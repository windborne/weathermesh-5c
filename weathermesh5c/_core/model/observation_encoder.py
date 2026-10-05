import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..meshes import GPSROData
from .config import ModelConfig
from .cross_attention import CrossTransformer
from .embeddings import sincos_spatial_embed
from .observation_utils import (
    ResidualMLP,
    expand_observation_latent,
    flat_observation_latent_shape,
    locate_horizontal,
    locate_observations,
    locate_pressure,
    normalize_observations,
    normalize_relative_time,
    observation_latent_shape,
)

logger = logging.getLogger(__name__)


def get_latent_mesh(config):
    if config.hlmesh.depth != 5:
        raise ValueError(
            f"WeatherMesh-5c requires a depth-5 latent mesh, got {config.hlmesh.depth}"
        )
    return config.hlmesh


@dataclass
class ObservationEncoderConfig:
    latent_size: int = 256
    attention_depth: int = 4
    attention_head_size: int = 32
    background_tokens: int = 8
    token_budget: int = 2**15
    position_size: int = 16
    profile_latent_size: int = 128
    profile_group_size: int = 12


ENCODER_CONFIG_MAIN = ObservationEncoderConfig(
    latent_size=512,
    attention_depth=2,
    attention_head_size=32,
    background_tokens=8,
    token_budget=2**15,
    position_size=16,
)

ENCODER_CONFIG_GPSRO_MAIN = ObservationEncoderConfig(
    latent_size=512,
    attention_depth=2,
    attention_head_size=32,
    background_tokens=8,
    token_budget=2**15,
    position_size=16,
    profile_latent_size=128,
)


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        meshes,
        config: ModelConfig,
        enc_config: ObservationEncoderConfig,
    ):
        super().__init__()
        self.meshes = meshes

        self.config = config
        self.enc_config = enc_config

        self._build_mesh_projections()
        self.tr = ObservationCrossTransformer(
            self.mesh_geometry,
            self.config,
            self.enc_config,
        )

        self.type_embedding = nn.Parameter(torch.randn(self.config.assimilation_latent_size))

        self.string_id = "".join([mesh.string_id for mesh in self.meshes])

    def _embedding_size(self, mesh):
        return len(mesh.full_varlist) + (len(mesh.indices) * self.enc_config.position_size)

    def _build_mesh_projections(self):
        self.mesh_geometry = self.meshes[0].observation_geometry
        assert set(mesh.observation_geometry for mesh in self.meshes) == {self.mesh_geometry}, (
            f"All meshes must have the same observation_geometry {self.mesh_geometry} vs {set(mesh.observation_geometry for mesh in self.meshes)}"
        )

        all_Cobs = [len(mesh.full_varlist) for mesh in self.meshes]
        assert len(set(all_Cobs)) == len(all_Cobs), (
            f"All meshes must have a unique number of observation channels (since that's how we decide which MLP to use) {all_Cobs} where {set(all_Cobs)} so {len(set(all_Cobs))} != {len(all_Cobs)}"
        )
        assert self.enc_config.latent_size >= max(all_Cobs), (
            f"latent_size {self.enc_config.latent_size} must be at least the maximum "
            f"observation channel count {max(all_Cobs)}"
        )

        self.mesh_mlps = nn.ModuleDict(
            {
                str(len(mesh.full_varlist)): ResidualMLP(
                    self._embedding_size(mesh),
                    self.enc_config.latent_size,
                )
                for mesh in self.meshes
            }
        )

        self.mesh_map = {len(mesh.full_varlist): mesh for mesh in self.meshes}

    def _coordinates(self, x, mesh):
        if mesh.observation_geometry == "3d_pr":
            assert "pres_hpa" in mesh.full_varlist, (
                f"Mesh {mesh.string_id} must have pres_hpa {mesh.full_varlist}"
            )
            pressure = x[:, mesh.full_varlist.index("pres_hpa")].clone().contiguous()
        else:
            pressure = torch.full_like(x[:, 0], torch.nan)

        lat = x[:, mesh.full_varlist.index("lat_deg")]
        lon = x[:, mesh.full_varlist.index("lon_deg")]

        relative_time = x[:, mesh.full_varlist.index("reltime_hours")]

        return pressure, lat, lon, relative_time

    def _locate(self, observations, mesh, device):
        pressure, latitude, longitude, relative_time = self._coordinates(observations, mesh)

        indices, coordinates, valid = locate_observations(
            pressure,
            latitude,
            longitude,
            relative_time,
            mesh.observation_geometry,
            self.config,
            mesh.da_window_size,
            device,
            get_latent_mesh(self.config),
        )

        if logger.isEnabledFor(logging.DEBUG):
            n_invalid = (~valid).sum().item()
            if n_invalid:
                logger.debug("Skipped %d observations with invalid coordinates", n_invalid)
        observations = observations[valid]
        indices = [index[valid] for index in indices]
        coordinates = [coordinate[valid] for coordinate in coordinates]

        latent_indices = torch.stack(indices, dim=-1)

        return observations, latent_indices, torch.stack(coordinates, dim=-1)

    def empty_latent(self, device):
        dtype = torch.float16
        hlmesh = get_latent_mesh(self.config)
        B, D, Nvert, C = flat_observation_latent_shape(
            self.config, self.meshes[0].observation_geometry, hlmesh
        )
        latent = torch.zeros([B, D, Nvert, C], dtype=dtype, device=device)
        return latent + self.type_embedding.to(dtype)

    def _embed(self, observations, mesh, coordinates):
        position = sincos_spatial_embed(coordinates, embed_dim=self.enc_config.position_size)
        observations = normalize_observations(observations, mesh)
        embedded = torch.cat([observations, position], dim=-1).to(torch.float16)
        assert -11 < embedded.min() and embedded.max() < 11, (
            f"Observation value is out of range: {embedded}"
        )
        return embedded

    def forward(self, observation_batches):
        encoded_batches = []
        index_batches = []
        for observations in observation_batches:
            observations = observations.clone()
            channel_count = observations.shape[1]
            assert channel_count in self.mesh_map, (
                f"Input tensor must be a recognized input mesh {channel_count} vs "
                f"{self.mesh_map.keys()}"
            )

            mesh = self.mesh_map[channel_count]
            observations, latent_indices, coordinates = self._locate(
                observations, mesh, observations.device
            )
            if observations.shape[0]:
                embedded = self._embed(observations, mesh, coordinates)
                encoded_batches.append(self.mesh_mlps[str(channel_count)](embedded))
                index_batches.append(latent_indices)

        if not encoded_batches:
            logger.debug("No valid observations for mesh %s", self.meshes[0].string_id)
            return self.empty_latent(observation_batches[0].device)

        encoded = torch.cat(encoded_batches, dim=0)
        latent_indices = torch.cat(index_batches, dim=0)
        assert latent_indices.shape[0] == encoded.shape[0]
        encoded = self.tr(encoded, latent_indices)
        return encoded + self.type_embedding.to(encoded.dtype)


class ObservationCrossTransformer(nn.Module):
    def __init__(self, observation_geometry, config, enc_config):
        super().__init__()
        self.observation_geometry = observation_geometry

        self.config = config
        self.enc_config = enc_config
        _, self.depth, self.vertex_count, self.channel_count = observation_latent_shape(
            config, observation_geometry, get_latent_mesh(config)
        )
        self.bg_tok_embed = nn.Parameter(
            torch.randn(self.enc_config.background_tokens, self.enc_config.latent_size)
        )
        self.bg_to_latent = nn.Linear(
            self.enc_config.background_tokens * self.enc_config.latent_size,
            self.channel_count,
        )
        self.ob_to_bg_transformer = CrossTransformer(
            self.enc_config.latent_size,
            depth=self.enc_config.attention_depth,
            dim_head=self.enc_config.attention_head_size,
        )

    def forward(self, observations, latent_indices):
        _, channels = observations.shape
        assert self.vertex_count == get_latent_mesh(self.config).vertices.shape[0], (
            f"Unexpected number of vertices {self.vertex_count} vs "
            f"{get_latent_mesh(self.config).vertices.shape[0]}"
        )
        device = observations.device
        dtype = observations.dtype
        assert dtype == torch.float16, f"Observation dtype is {dtype}"

        latent_space = torch.zeros(
            [1, self.vertex_count, self.depth, self.channel_count],
            device=device,
            dtype=dtype,
        )

        assert channels == self.enc_config.latent_size, (
            f"Input tensor has {channels} features; expected {self.enc_config.latent_size}"
        )
        assert latent_indices.shape[1] == 2, (
            f"Expected pressure and vertex indices, got {latent_indices.shape[1]} columns"
        )

        flat_indices = latent_indices[:, 0] * self.vertex_count + latent_indices[:, 1]

        unique, inverse_indices, counts = torch.unique(
            flat_indices, return_counts=True, return_inverse=True
        )

        starts = torch.zeros(len(unique), device=device, dtype=torch.long)
        starts[1:] = torch.cumsum(counts[:-1], 0)
        sorted_indices = torch.argsort(inverse_indices).to(device=device)
        del inverse_indices

        counts_sorted = counts.sort(descending=True)

        current_bucket = 0
        num_unique = unique.shape[0]
        while current_bucket < num_unique:
            bucket_max_count = counts_sorted.values[current_bucket].item()

            batch_capacity = self.enc_config.token_budget // bucket_max_count
            batch_end = min(current_bucket + batch_capacity, num_unique)
            batch = unique[counts_sorted.indices[current_bucket:batch_end]]
            batch_size = len(batch)

            background = torch.zeros(
                batch_size,
                self.enc_config.background_tokens,
                self.enc_config.latent_size,
                device=device,
                dtype=dtype,
            )
            background[:] = self.bg_tok_embed

            def build_context(
                current_bucket=current_bucket,
                batch_end=batch_end,
                bucket_max_count=bucket_max_count,
                batch_size=batch_size,
            ):
                batch_indices = counts_sorted.indices[current_bucket:batch_end]

                batch_starts = starts[batch_indices]
                batch_counts = counts[batch_indices]
                del batch_indices

                max_count = batch_counts.max().item()
                assert bucket_max_count == max_count, (
                    f"bucket_max_count {bucket_max_count} != max_count {max_count}"
                )

                column_indices = (
                    torch.arange(max_count, device=device).unsqueeze(0).expand(batch_size, -1)
                )
                mask = column_indices < batch_counts.unsqueeze(1)
                del batch_counts

                indices_into_sorted = batch_starts.unsqueeze(1) + column_indices
                del batch_starts
                del column_indices
                safe_indices = torch.where(
                    mask, indices_into_sorted, torch.zeros_like(indices_into_sorted)
                )
                del indices_into_sorted

                values = observations[sorted_indices[safe_indices]]
                del safe_indices
                context = values * mask.unsqueeze(-1)
                del values

                context_mask = ~(context == 0).all(dim=-1)
                context_mask = torch.where(context_mask, 0.0, float("-inf"))
                del mask

                return context, context_mask

            context, context_mask = build_context()
            output = self.ob_to_bg_transformer(background, context, context_mask)

            del background, context, context_mask

            vertex_indices = batch % self.vertex_count
            pressure_indices = batch // self.vertex_count
            del batch
            latent_space[0, vertex_indices, pressure_indices] = self.bg_to_latent(
                output[:].flatten(1)
            ).to(dtype=latent_space.dtype)
            del output
            del vertex_indices, pressure_indices

            current_bucket += batch_size

        latent_space = latent_space.view(
            1, self.vertex_count, self.depth, self.channel_count
        ).permute(0, 2, 1, 3)
        latent_space = expand_observation_latent(
            latent_space, self.config, self.observation_geometry, get_latent_mesh(self.config)
        )
        assert latent_space.dtype == torch.float16

        assert latent_space.shape[2] == self.vertex_count
        return latent_space


class GPSROEncoder(nn.Module):
    def __init__(self, meshes, config, enc_config):
        super().__init__()
        self.meshes = meshes
        self.config = config
        self.enc_config = enc_config
        assert isinstance(meshes[0], GPSROData), "GPSROEncoder requires GPSROData"
        assert len(meshes) == 1, "GPSROEncoder requires exactly one mesh"
        self.input_dim = len(self.meshes[0].full_varlist)

        assert self.enc_config.latent_size % self.enc_config.attention_head_size == 0, (
            f"latent_size {self.enc_config.latent_size} must be divisible by attention_head_size {self.enc_config.attention_head_size}"
        )
        self.profile_mlp = ResidualMLP(
            self.input_dim + self._position_size() + 1,
            self.enc_config.profile_latent_size,
        )

        self.ob_conv = nn.Linear(
            self.enc_config.profile_group_size * self.enc_config.profile_latent_size,
            self.enc_config.latent_size,
        )

        self.ob_tr = ObservationCrossTransformer(
            self.meshes[0].observation_geometry,
            self.config,
            self.enc_config,
        )

        self.type_embedding = nn.Parameter(torch.randn(self.config.assimilation_latent_size))

        self.string_id = "".join([mesh.string_id for mesh in self.meshes])

    def _position_size(self):
        return 4 * self.enc_config.position_size

    def _pressure_levels(self, device):
        altitude = torch.arange(0.0, 30_000.0, 100.0, device=device)
        pressure = torch.empty_like(altitude)

        g = 9.80665  # m s⁻²
        R_air = 287.05287  # J kg⁻¹ K⁻¹

        T0, L0, p0 = 288.15, -0.0065, 1013.25  # K, K/m, hPa
        layer = altitude <= 11_000.0
        pressure[layer] = p0 * (1 + L0 * altitude[layer] / T0) ** (-g / (R_air * L0))

        T1, h1, p1 = 216.65, 11_000.0, 226.3210
        layer = (altitude > 11_000.0) & (altitude <= 20_000.0)
        pressure[layer] = p1 * torch.exp(-g * (altitude[layer] - h1) / (R_air * T1))

        T2, L2, h2, p2 = 216.65, 0.001, 20_000.0, 54.7489
        layer = altitude > 20_000.0
        temperature = T2 + L2 * (altitude[layer] - h2)
        pressure[layer] = p2 * (temperature / T2) ** (-g / (R_air * L2))

        return pressure

    def _normalize(self, x, mesh):
        if "lat_deg" in mesh.full_varlist:
            x[:, :, mesh.full_varlist.index("lat_deg")] /= 90.0
        if "lon_deg" in mesh.full_varlist:
            x[:, :, mesh.full_varlist.index("lon_deg")] /= 180.0
        if "pres_hpa" in mesh.full_varlist:
            x[:, :, mesh.full_varlist.index("pres_hpa")] /= 1000.0
        minimum, maximum = mesh.da_window_size
        max_window = np.max([np.abs(minimum), np.abs(maximum)])
        if "reltime_hours" in mesh.full_varlist:
            x[:, :, mesh.full_varlist.index("reltime_hours")] /= max_window
        return x

    def _embed(self, profiles, coordinates, mesh):
        profile_count, altitude_count, channel_count = profiles.shape
        position = sincos_spatial_embed(
            coordinates,
            embed_dim=self.enc_config.position_size,
        )
        profiles = self._normalize(profiles, mesh)
        position = position.view(profile_count, altitude_count, self._position_size())
        embedded = torch.cat([profiles, position], dim=-1).to(torch.float16)

        assert embedded.shape == (
            profile_count,
            altitude_count,
            channel_count + self._position_size(),
        ), f"Embedded GPSRO shape is {embedded.shape}"
        assert -11 < embedded.min() and embedded.max() < 11, (
            f"Observation value is out of range: {embedded}"
        )
        return embedded

    def empty_latent(self, device):
        dtype = torch.float16
        B, D, Nvert, C = flat_observation_latent_shape(
            self.config, self.meshes[0].observation_geometry, get_latent_mesh(self.config)
        )
        latent = torch.zeros([B, D, Nvert, C], dtype=dtype, device=device)
        return latent + self.type_embedding.to(dtype)

    def forward(self, x):
        x = x[0]

        profile_count, altitude_count, channel_count = x.shape
        device = x.device
        assert altitude_count == 300, (
            f"GPSROEncoder requires 300 altitude features but got {altitude_count}"
        )
        assert channel_count == self.input_dim, (
            f"GPSROEncoder requires {self.input_dim} observation channels but got {channel_count}"
        )

        coordinates = []
        indices = []
        pressure = self._pressure_levels(device)
        pressure_coordinate, pressure_index, _ = locate_pressure(
            pressure, self.config, self.meshes[0].observation_geometry, device
        )
        latitude = x[:, :, self.meshes[0].full_varlist.index("lat_deg")].reshape(-1)
        longitude = x[:, :, self.meshes[0].full_varlist.index("lon_deg")].reshape(-1)
        latitude_coordinate, longitude_coordinate, vertex_index = locate_horizontal(
            latitude, longitude, get_latent_mesh(self.config), device
        )
        pressure_coordinate = pressure_coordinate.expand(profile_count, -1).reshape(-1)
        pressure_index = pressure_index.expand(profile_count, -1).reshape(-1)
        coordinates.extend([pressure_coordinate, latitude_coordinate, longitude_coordinate])
        indices.extend([pressure_index, vertex_index])
        latent_indices = torch.stack(indices, dim=-1)
        group_size = self.enc_config.profile_group_size
        latent_indices = latent_indices[group_size // 2 :: group_size]

        reltime = x[:, :, self.meshes[0].full_varlist.index("reltime_hours")].reshape(-1)
        coordinates.append(normalize_relative_time(reltime, self.meshes[0].da_window_size))
        x = self._embed(x, torch.stack(coordinates, dim=-1), self.meshes[0])

        altitude_embedding = torch.arange(300, device=x.device) / 300.0  # Normalize to 0-1
        altitude_embedding = altitude_embedding.view(1, 300, 1)
        altitude_embedding = altitude_embedding.expand(profile_count, -1, 1)

        x = torch.cat([x, altitude_embedding], dim=-1)
        x = x.to(torch.float16)
        x = self.profile_mlp(x)

        x = self.ob_conv(
            x.view(
                profile_count,
                300 // self.enc_config.profile_group_size,
                self.enc_config.profile_group_size * self.enc_config.profile_latent_size,
            )
        )
        x = x.view(x.shape[0] * x.shape[1], self.enc_config.latent_size)

        x = self.ob_tr(x, latent_indices)
        x = x + self.type_embedding.to(x.dtype)
        return x
