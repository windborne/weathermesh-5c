import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from .config import ModelConfig
from .heal import LatentMesh
from .healpix_transformer import HealpixBlock
from .transformer import WideTransformer

logger = logging.getLogger(__name__)


@dataclass
class AssimilationConfig:
    """Fixed WeatherMesh-5c data-assimilation architecture."""

    assimilation_depth: int = 8
    assimilation_vertex_chunks: int = 2
    assimilation_latent_size: int = 768
    assimilation_hours: list[int] = field(default_factory=lambda: [-6])
    observation_mix_repeats: int = 3
    observation_mix_depth: int = 2
    observation_attention_size: int = 1536
    observation_feedforward_size: int = 1536
    observation_head_size: int = 24
    vertical_mix_depth: int = 1
    vertical_attention_size: int = 4096
    vertical_feedforward_size: int = 1536
    vertical_head_size: int = 64
    spatial_attention_size: int = 3072
    spatial_heads: int = 48
    spatial_mlp_ratio: int = 2


class WeatherMeshModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        analysis_encoders,
        observation_encoders,
        da_transformer,
        processors,
        predecoder,
        decoders,
        forker,
    ):
        super().__init__()
        self.config = config
        self.anl_encoders = nn.ModuleList(analysis_encoders)
        self.obs_encoders = nn.ModuleList(observation_encoders)
        self.da_transformer = da_transformer
        self.processors = nn.ModuleDict(processors)
        self.predecoder = predecoder
        self.decoders = nn.ModuleList(decoders)
        self.forker = forker

        if len(self.anl_encoders) != 1:
            raise ValueError("WeatherMesh-5c requires one background encoder")

        observation_ids = [encoder.string_id for encoder in self.obs_encoders]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError(f"Observation encoder IDs must be unique: {observation_ids}")
        self.da_transformer.initialize_tr0ds(observation_ids)

        assert self.config.latent_size % self.config.assimilation_latent_size == 0, (
            f"latent_size must be divisible by assimilation_latent_size, {self.config.latent_size} % {self.config.assimilation_latent_size} != 0"
        )
        assert len(self.config.processor_dts) == 1, (
            f"WeatherMesh-5c requires one processor, got {len(self.config.processor_dts)}"
        )

        self.background_tokens_per_level = (
            self.config.latent_size // self.config.assimilation_latent_size
        )
        self.scratch_space = nn.Parameter(
            torch.randn(6 * self.background_tokens_per_level, self.config.assimilation_latent_size)
        )

        self.step_hours = self.config.processor_dts[0]

    def move_assets(self, device):
        self.config.move_over(device)

    def _advance(self, state, hours, start_hour):
        processor = self.processors[str(self.step_hours)]

        while hours > 0:
            state = processor(state)
            logger.debug(
                "Advanced analysis by %d hours (%d -> %d)",
                self.step_hours,
                hours + start_hour,
                hours + start_hour - self.step_hours,
            )
            hours -= self.step_hours

        return state

    def _latent_levels(self, mesh):
        pressure_levels = len(self.config.latent_levels)
        if mesh.observation_geometry == "2d_pr+sfc":
            return list(range(pressure_levels + 1))
        if mesh.observation_geometry == "3d_pr":
            return list(range(pressure_levels))
        if mesh.observation_geometry == "2d_sfc":
            return [pressure_levels]
        raise ValueError(f"Unknown observation geometry: {mesh.observation_geometry}")

    def _encode_observations(self, cycle):
        latents = []
        encoder_ids = []
        latent_levels = []

        for encoder in self.obs_encoders:
            observations = []
            for mesh in encoder.meshes:
                values = cycle[mesh.mesh_id]
                if values.numel():
                    observations.append(values[0])
                else:
                    logger.debug("No data available for %s", mesh.string_id)

            if observations:
                latent = encoder(observations)
            else:
                logger.debug("Using an empty latent for %s", encoder.string_id)
                latent = encoder.empty_latent(cycle[encoder.meshes[0].mesh_id].device)

            latents.append(latent.permute(0, 2, 1, 3))
            encoder_ids.append(encoder.string_id)
            latent_levels.append(self._latent_levels(encoder.meshes[0]))

        return latents, encoder_ids, latent_levels

    def assimilate(self, cycles, *, first_hour=None):
        background_hours = [hour for hour, cycle in cycles.items() if cycle["has_analysis"]]
        if len(background_hours) != 1:
            raise ValueError(f"Expected one background analysis, got {background_hours}")
        background_hour = background_hours[0]

        if first_hour is None:
            possible_hours = [hour for hour in self.config.assimilation_hours if hour > background_hour]
            if not possible_hours:
                raise ValueError(f"No assimilation hour follows background hour {background_hour}")
            first_hour = possible_hours[0]
        if first_hour <= background_hour:
            raise ValueError("The first assimilation must follow the background")
        if first_hour not in cycles or not cycles[first_hour]["has_observations"]:
            raise ValueError(f"Assimilation hour {first_hour} has no observations")

        assimilation_hours = [hour for hour in cycles if hour >= first_hour]
        if not all(cycles[hour]["has_observations"] for hour in assimilation_hours):
            raise ValueError("Every assimilation cycle must contain observations")

        background_cycle = cycles[background_hour]
        background = background_cycle[self.anl_encoders[0].mesh.mesh_id]
        state = self.anl_encoders[0](background, background_cycle["timestamp"])
        state = self._advance(
            state,
            first_hour - background_hour,
            -first_hour,
        )

        scratch = self.scratch_space.to(torch.float16).unsqueeze(0).unsqueeze(1)
        scratch = scratch.expand(*state.shape)
        if scratch.shape != state.shape:
            raise ValueError(f"Scratch shape {scratch.shape} does not match state {state.shape}")

        state_levels = [
            list(range(len(self.config.latent_levels) + 1))
            for _ in range(self.background_tokens_per_level)
        ]
        assimilation_state = {
            "analysis": state,
            "analysis_levels": state_levels,
            "scratch": scratch,
            "scratch_levels": state_levels,
        }

        for hour in assimilation_hours:
            observations, encoder_ids, observation_levels = self._encode_observations(cycles[hour])
            assimilation_state["observations"] = observations
            assimilation_state["observation_levels"] = observation_levels
            logger.debug("Assimilating observations at hour %d", hour)
            assimilation_state = self.da_transformer(assimilation_state, encoder_ids)

            if hour < 0:
                assimilation_state["analysis"] = self._advance(
                    assimilation_state["analysis"], self.step_hours, -hour - self.step_hours
                )

        return assimilation_state["analysis"], background_hour


class AssimilationTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.tr1ds = nn.ModuleList(
            [
                WideTransformer(
                    dim=self.config.assimilation_latent_size,
                    depth=self.config.vertical_mix_depth,
                    attention_hidden_dim=self.config.vertical_attention_size,
                    dim_head=self.config.vertical_head_size,
                    feedforward_hidden_dim=self.config.vertical_feedforward_size,
                )
                for _ in range(self.config.assimilation_depth)
            ]
        )

        self.tr3ds = nn.ModuleList(
            [
                HealpixBlock(
                    dim=self.config.assimilation_latent_size,
                    num_heads=self.config.spatial_heads,
                    hlmesh=None,
                    mlp_ratio=self.config.spatial_mlp_ratio,
                    hidden_dim=self.config.spatial_attention_size,
                )
                for _ in range(self.config.assimilation_depth)
            ]
        )

        self._mesh_cache = {}
        self._position_cache = {}

    def initialize_tr0ds(self, tr0d_indices):
        self.tr0ds = nn.ModuleDict(
            {
                encoder_id: WideTransformer(
                    dim=self.config.assimilation_latent_size,
                    depth=self.config.observation_mix_depth,
                    attention_hidden_dim=self.config.observation_attention_size,
                    dim_head=self.config.observation_head_size,
                    feedforward_hidden_dim=self.config.observation_feedforward_size,
                )
                for encoder_id in tr0d_indices
            }
        )

    def _mesh_for(self, level_groups):
        cache_key = tuple(tuple(group) for group in level_groups)

        if cache_key not in self._mesh_cache:
            mesh = LatentMesh(
                depth=5,
                D=level_groups,
                KL=9,
                KD=3,
                B=self.config.hlmesh.B,
                dim=self.channels,
                tri=self.config.hlmesh.tri,
                bundle=self.config.bundle,
            )
            del mesh.vertices
            mesh = mesh.to(self.device)
            if mesh.Nvert != self.vertices:
                raise ValueError(f"Mesh has {mesh.Nvert} vertices, expected {self.vertices}")
            self._mesh_cache[cache_key] = mesh

        return self._mesh_cache[cache_key]

    def _add_position_embeddings(self, state):
        all_levels = (
            state["analysis_levels"] + state["scratch_levels"] + state["observation_levels"]
        )
        mesh = self._mesh_for(all_levels)

        def add(latent, level_groups):
            latent = latent.contiguous()
            start = 0
            for levels in level_groups:
                key = tuple(levels)
                if key not in self._position_cache:
                    self._position_cache[key] = mesh.posemb_mesh(
                        mesh.lonlat,
                        len(levels),
                        self.channels,
                        T1=300,
                        T2=10,
                        dtype=torch.float16,
                    )[None].to(self.device)
                length = len(levels)
                latent[:, :, start : start + length] += self._position_cache[key]
                start += length
            return latent

        state["analysis"] = add(state["analysis"], state["analysis_levels"])
        state["scratch"] = add(state["scratch"], state["scratch_levels"])
        for index, latent in enumerate(state["observations"]):
            state["observations"][index] = add(latent, [state["observation_levels"][index]])
        return state

    def _transform_chunks(self, latent, transformer):
        if self.vertices % self.config.assimilation_vertex_chunks:
            raise ValueError("The assimilation vertex count must divide into equal chunks")
        has_batch = latent.ndim == 4
        if has_batch:
            if latent.shape[0] != self.batch_size:
                raise ValueError(f"Unexpected assimilation batch size: {latent.shape[0]}")
            latent = latent[0]

        chunk_size = self.vertices // self.config.assimilation_vertex_chunks
        for start in range(0, self.vertices, chunk_size):
            latent[start : start + chunk_size] = transformer(latent[start : start + chunk_size])
        return latent.unsqueeze(0) if has_batch else latent

    @staticmethod
    def _matching_levels(latent, latent_levels, observation_levels):
        if len(latent_levels) != 1:
            raise ValueError(f"Expected one latent-level group, got {latent_levels}")
        level_indices = {level: index for index, level in enumerate(latent_levels[0])}
        indices = [level_indices[level] for level in observation_levels if level in level_indices]
        return latent[:, :, indices]

    def _mix_observations(self, state, encoder_ids):
        if len(state["observations"]) != len(encoder_ids):
            raise ValueError("Each observation latent needs an encoder ID")
        for index, observation in enumerate(state["observations"]):
            model_latents = [
                self._matching_levels(
                    state[name], state[f"{name}_levels"], state["observation_levels"][index]
                )
                for name in ("analysis", "scratch")
            ]
            model_length = sum(latent.shape[2] for latent in model_latents)
            combined = torch.cat((*model_latents, observation), dim=2)
            combined = self._transform_chunks(combined, self.tr0ds[encoder_ids[index]])
            state["observations"][index] = combined[:, :, model_length:].clone(
                memory_format=torch.contiguous_format
            )
        return state

    def _mix_vertical(self, state, layer):
        combined = torch.cat([state["analysis"], state["scratch"], *state["observations"]], dim=2)
        combined = self._transform_chunks(combined, self.tr1ds[layer])
        start = 0
        for name in ("analysis", "scratch"):
            length = sum(len(levels) for levels in state[f"{name}_levels"])
            state[name] = combined[:, :, start : start + length].clone(
                memory_format=torch.contiguous_format
            )
            start += length
        for index, levels in enumerate(state["observation_levels"]):
            length = len(levels)
            state["observations"][index] = combined[:, :, start : start + length].clone(
                memory_format=torch.contiguous_format
            )
            start += length
        return state

    def _mix_spatial(self, state, layer):
        transformer = self.tr3ds[layer]
        transformer.attn.hlmesh = self._mesh_for(state["analysis_levels"] + state["scratch_levels"])
        analysis_length = state["analysis"].shape[2]
        combined = transformer(torch.cat([state["analysis"], state["scratch"]], dim=2))
        state["analysis"] = combined[:, :, :analysis_length]
        state["scratch"] = combined[:, :, analysis_length:]
        return state

    @staticmethod
    def _check_dtype(state):
        latents = [state["analysis"], state["scratch"], *state["observations"]]
        if any(latent.dtype != torch.float16 for latent in latents):
            raise TypeError("Assimilation latents must use float16")

    def forward(self, state, encoder_ids):
        assert hasattr(self, "tr0ds"), "tr0ds must be initialized before calling forward"
        self.batch_size, self.vertices, _, self.channels = state["analysis"].shape
        self.device = state["analysis"].device
        self._check_dtype(state)
        state = self._add_position_embeddings(state)

        for _ in range(self.config.observation_mix_repeats):
            state = self._mix_observations(state, encoder_ids)
            self._check_dtype(state)

        for layer in range(self.config.assimilation_depth):
            state = self._mix_vertical(state, layer)
            self._check_dtype(state)
            state = self._mix_spatial(state, layer)
            self._check_dtype(state)

        return state
