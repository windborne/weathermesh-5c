from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .assets import RuntimeBundle
from .background.download import OPEN_LEVELS
from .background.prepare import OpenIFSFields, build_background, remap_oisst
from .checkpoint import load_model
from .ensemble import decode_products_for_transfer
from .ingest._realchans import real_chans
from .ingest.prepare import MESH_SOURCES, build_batch, prepare_observations
from .ingest.storage import _deduplicate
from .observations import pack_observations
from .output import ProductTransform, aggregate_decoded_products, physicalize_statistics

SURFACE_VARIABLES = (
    "u10m",
    "v10m",
    "t2m",
    "msl",
    "d2m",
    "u100m",
    "v100m",
    "snswrf:mean:1h",
    "tcc",
    "tp:sum:3h",
    "t2m:max:3h",
    "t2m:min:3h",
    "ws10m",
    "ws100m",
)
PRESSURE_INPUTS = {"z": "129_z", "t": "130_t", "u": "131_u", "v": "132_v", "q": "133_q"}
SURFACE_INPUTS = {
    "u10m": "165_10u",
    "v10m": "166_10v",
    "t2m": "167_2t",
    "msl": "151_msl",
    "d2m": "168_2d",
    "u100m": "246_100u",
    "v100m": "247_100v",
    "skt": "skt",
    "lsm": "lsm",
}
IASI_CHANNELS = tuple(real_chans)


class Inference(torch.nn.Module):
    """Stable numerical interface shared by standalone and external wrappers.

    Field preparation and physicalization retain the native CPU arithmetic.
    All latent operations own their results and leave borrowed tensors unchanged.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._model = model
        self.register_buffer("_device", torch.empty(0), persistent=False)
        self._transform = ProductTransform.from_model(model)
        mesh = model.anl_encoders[0].mesh
        self.input_latitudes = np.asarray(mesh.lats)
        self.input_longitudes = np.mod(mesh.lons, 360)
        self.input_variables = tuple(
            f"{name}{level}" for name in PRESSURE_INPUTS for level in OPEN_LEVELS
        ) + tuple(SURFACE_INPUTS)
        mesh = model.decoders[0].mesh
        self.output_latitudes = np.asarray(mesh.lats)
        self.output_longitudes = np.mod(mesh.lons, 360)
        self.pressure_variables = tuple(
            f"{name}{int(level)}"
            for name in ("z", "t", "u", "v", "q")
            for level in mesh.levels
        )
        self.variables = (*self.pressure_variables, *SURFACE_VARIABLES)
        self.instantaneous_variables = tuple(v for v in self.variables if ":" not in v)
        self.observation_sources = tuple(
            MESH_SOURCES[mesh.string_id]
            for encoder in model.obs_encoders
            for mesh in encoder.meshes
        )

    @classmethod
    def load(cls, resolve: Callable[[str], str | Path]) -> Inference:
        """Load resolved package files directly, with no second asset cache."""
        manifest = Path(resolve("manifest.json"))
        return cls(load_model(RuntimeBundle(manifest.parent, resolve=resolve)))

    @property
    def device(self) -> torch.device:
        """Device used by the numerical model."""
        return self._device.device

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self._model.move_assets(self.device)
        return self

    @contextmanager
    def _autocast(self):
        with torch.autocast(
            self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
        ):
            yield

    @staticmethod
    def prepare_background(fields: OpenIFSFields, sst: np.ndarray | None) -> np.ndarray:
        """Apply the trained normalization, interpolation, masking and rounding."""
        return build_background(fields, oisst=sst)

    @staticmethod
    def prepare_sst(
        values: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray
    ) -> np.ndarray:
        """Map physical SST to the encoder grid with native interpolation."""
        return remap_oisst(values, latitudes, longitudes)

    def prepare_observations(
        self,
        source: str,
        hours: Iterable[tuple[datetime, np.ndarray]],
        valid_time: datetime,
        *,
        max_observations: int | None = None,
    ) -> torch.Tensor:
        """Apply native QC, time windows and report-level sampling to one source."""
        mesh = next(
            (
                mesh
                for encoder in self._model.obs_encoders
                for mesh in encoder.meshes
                if MESH_SOURCES[mesh.string_id] == source
            ),
            None,
        )
        if mesh is None:
            raise ValueError(f"Unsupported observation source: {source}")
        return prepare_observations(
            mesh, hours, valid_time, max_observations=max_observations
        ).to(self.device)

    def prepare_raw_observations(
        self,
        source: str,
        fields: Mapping[str, np.ndarray],
        valid_time: datetime,
        *,
        max_observations: int | None = None,
    ) -> torch.Tensor:
        """Prepare physical observation columns without a BUFR or cache dependency."""
        return self.prepare_observations(
            source,
            self.pack_raw_observations(source, fields),
            valid_time,
            max_observations=max_observations,
        )

    @staticmethod
    def pack_raw_observations(
        source: str,
        fields: Mapping[str, np.ndarray] | Iterable[Mapping[str, np.ndarray]],
    ) -> list[tuple[datetime, np.ndarray]]:
        """Apply native report preparation once, before update-window sampling.

        Accepts one column mapping or bounded batches. Merge compact arrays in
        report order, with the same hourly deduplication as the native store.
        No sampling or RNG state is used here.
        """
        batches = (fields,) if isinstance(fields, Mapping) else fields
        hours: dict[datetime, list[np.ndarray]] = {}
        for batch in batches:
            for hour, values in pack_observations(source, batch):
                if len(values):
                    hours.setdefault(hour, []).append(values)
        result = []
        for hour in sorted(hours):
            values = np.concatenate(hours[hour])
            values, _ = _deduplicate(values, np.zeros(len(values)))
            result.append((hour, values))
        return result

    @torch.inference_mode()
    def encode(
        self, background: np.ndarray | torch.Tensor, time: datetime
    ) -> torch.Tensor:
        """Encode one prepared physical background without advancing it."""
        values = torch.as_tensor(background, device=self.device).clone()
        if values.ndim == 3:
            values = values.unsqueeze(0)
        timestamp = torch.tensor(
            [time.timestamp()], dtype=torch.float64, device=self.device
        )
        with self._autocast():
            return self._model.anl_encoders[0](values, timestamp)

    @torch.inference_mode()
    def assimilate(
        self,
        background: np.ndarray | torch.Tensor,
        time: datetime,
        observations: Mapping[int, Mapping[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Return the native AIDA latent at analysis time from two or three updates."""
        hours = sorted(observations)
        if hours not in ([-3, 0], [-6, -3, 0]):
            raise ValueError("AIDA requires two or three successive three-hour updates")
        meshes = {
            MESH_SOURCES[m.string_id]: m
            for e in self._model.obs_encoders
            for m in e.meshes
        }
        cycles = {
            hour: {
                mesh.mesh_id: observations[hour].get(
                    source, torch.empty((1, 0, 0), device=self.device)
                )
                for source, mesh in meshes.items()
            }
            for hour in hours
        }
        values = torch.as_tensor(background, device=self.device).clone()
        batch = build_batch(
            self._model,
            values,
            time,
            observations_root=".",
            analysis_offset_hours=hours[0] - 3,
            observation_cycles=cycles,
        )
        with self._autocast():
            latent, _ = self._model.assimilate(batch, first_hour=hours[0])
        return latent

    @torch.inference_mode()
    def step(self, latent: torch.Tensor) -> torch.Tensor:
        """Advance a borrowed latent by three hours."""
        with self._autocast():
            return self._model.processors["3"](latent.to(self.device).clone())

    @torch.inference_mode()
    def fork(self, latent: torch.Tensor, *, hour: int, seed: int) -> torch.Tensor:
        """Apply one native learned branch with an isolated random stream."""
        if hour < 12 or hour % 12:
            raise ValueError(
                "The learned fork requires a positive multiple of twelve hours"
            )
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), self._autocast():
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[self.device.index].manual_seed(seed)
            return self._model.forker(latent.to(self.device).clone(), 1, hour)

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor, *, pressure: bool = True) -> np.ndarray:
        """Return physical arrays in variables/latitude/longitude order.

        Native CPU arithmetic is intentional for exact output parity. Precipitation
        uses metres; pressure fields precede surface fields when requested.
        """
        frames = []
        for frame in latent:
            with self._autocast():
                decoded = decode_products_for_transfer(
                    self._model,
                    frame.unsqueeze(0).to(self.device).clone(),
                    include_pressure=pressure,
                )
            decoded = replace(
                decoded,
                surface=decoded.surface.cpu(),
                pressure=None if decoded.pressure is None else decoded.pressure.cpu(),
            )
            products = aggregate_decoded_products(decoded, self._transform)
            # Match the native accumulator's first update, including signed zero.
            for values in (products.surface, products.pressure):
                if values is not None:
                    np.add(values, np.float32(0), out=values)
            fields = physicalize_statistics(products, self._transform, "mean")
            surface = fields.surface
            surface[SURFACE_VARIABLES.index("tp:sum:3h")] /= np.float32(1000)
            frames.append(
                np.concatenate(
                    (fields.pressure.reshape(-1, *surface.shape[1:]), surface)
                )
                if pressure
                else surface
            )
        return np.stack(frames)
