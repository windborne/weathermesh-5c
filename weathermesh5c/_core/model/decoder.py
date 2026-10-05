import torch
from torch import nn

from ...assets import load_archive
from .heal import inference_compile
from .healpix_layers import HealpixUpsample
from .healpix_transformer import HealpixProcessor


class ForecastPredecoder(nn.Module):
    """Convert the depth-5 model state to global decoder features."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        if not hasattr(config, "hlmesh"):
            raise ValueError("ForecastPredecoder requires a HEALPix latent mesh")
        if config.hlmesh.depth != 5:
            raise ValueError("WeatherMesh-5c requires a depth-5 latent mesh")

        self.tr = HealpixProcessor(
            depth=config.encdec_tr_depth,
            num_heads=config.num_heads,
            dim=config.latent_size,
            hlmesh=config.hlmesh,
        )
        self.nvert = 5
        depth = config.hlmesh.depth
        self.latent_to_mid_sfc = HealpixUpsample(
            config.latent_size,
            config.sparse_mid_latent,
            depth,
            7,
        )
        self.latent_to_mid_pr = nn.ModuleList(
            HealpixUpsample(
                config.latent_size,
                config.sparse_mid_latent,
                depth,
                7,
            )
            for _ in range(self.nvert)
        )
        self.gelu = nn.GELU(approximate="tanh")
        self.lnorms = nn.ModuleList(
            nn.LayerNorm(config.sparse_mid_latent) for _ in range(self.nvert + 1)
        )

    def _project_level(self, x, index):
        projection = self.latent_to_mid_sfc if index == -1 else self.latent_to_mid_pr[index]
        x = projection(x)
        return self.gelu(self.lnorms[index](x).half()) + x

    def forward(self, x, sfc_only=False):
        x = self.tr(x)
        x_sfc = self._project_level(x[:, :, -1:, :].clone(), -1)
        if sfc_only:
            x_pr = [None] * self.nvert
        else:
            x_pr = [
                self._project_level(x[:, :, index : index + 1, :].clone(), index)
                for index in range(self.nvert)
            ]
        return [*x_pr, x_sfc]


@inference_compile
class GlobalForecastDecoder(nn.Module):
    """Decode global deterministic and quantile weather fields."""

    def __init__(
        self,
        mesh,
        config,
        n_quantiles=None,
        pr_quantile_vars=None,
        sfc_quantile_vars=None,
    ):
        super().__init__()
        self.config = config
        self.mesh = mesh
        self.use_upscaled = False

        self.factor = 4
        self.n_chunks = config.n_chunks_dec
        path = config.bundle.global_path(f"meshes/healpix_m2g_7_c{self.n_chunks}.npz")
        matrix_groups = load_archive(path)
        gathers = [
            torch.from_numpy(matrix_groups[f"gather_{index}"]) for index in range(self.n_chunks)
        ]
        subgathers = [
            torch.from_numpy(matrix_groups[f"subgather_{index}"]) for index in range(self.n_chunks)
        ]
        edge_features = [
            torch.from_numpy(matrix_groups[f"edge_features_{index}"])
            for index in range(self.n_chunks)
        ]
        self.n_neigh = subgathers[0].shape[2]
        self.n_edge_features = edge_features[0].shape[-1]
        if self.n_neigh != 4 or self.n_edge_features != 22:
            raise ValueError("The global decoder matrix contract changed")
        for gather, subgather in zip(gathers, subgathers, strict=True):
            if subgather.min() < 0 or subgather.max() >= gather.numel():
                raise ValueError("Decoder subgather index is outside the gathered grid")
        self.maximum_gather_index = max(int(gather.max()) for gather in gathers)

        def register_chunks(name, values):
            for index, value in enumerate(values):
                self.register_buffer(f"{name}{index}", value)

        register_chunks("gathers", gathers)
        register_chunks("subgathers", subgathers)
        register_chunks("edge_features", edge_features)

        self.n_quantiles = n_quantiles
        self.pr_quantile_vars = pr_quantile_vars
        self.sfc_quantile_vars = sfc_quantile_vars
        quantile_variables = pr_quantile_vars + sfc_quantile_vars
        self.sfc_det_vars = [
            variable for variable in mesh.sfc_vars if variable not in quantile_variables
        ]
        self.sublatent = config.dec_sublatent
        self.deconv_mlp_dim = config.deconv_mlp_dim
        gelu = nn.GELU(approximate="tanh")
        input_width = (self.sublatent + self.n_edge_features) * self.n_neigh
        self.prefinal_sfc = nn.Linear(input_width, self.deconv_mlp_dim)
        self.final_sfc = nn.Sequential(
            gelu,
            nn.Linear(self.deconv_mlp_dim, self.deconv_mlp_dim),
            gelu,
        )
        self.postfinal_sfc = nn.Linear(
            self.deconv_mlp_dim,
            len(self.sfc_det_vars) + len(sfc_quantile_vars),
        )
        self.prefinal_pr = nn.Linear(input_width, self.deconv_mlp_dim)
        self.final_pr = nn.Sequential(
            gelu,
            nn.Linear(self.deconv_mlp_dim, self.deconv_mlp_dim),
            gelu,
        )
        self.postfinal_pr = nn.Linear(
            self.deconv_mlp_dim,
            mesh.n_pr_vars * (config.patch_size[0] + n_quantiles),
        )

    def forward(self, x):
        pressure_variables = self.mesh.n_pr_vars
        batch = x[-1].shape[0]
        if pressure_variables != 5:
            raise ValueError("WeatherMesh-5c requires five pressure variables")
        if self.maximum_gather_index >= x[-1].shape[1]:
            raise ValueError("Surface decoder index is outside the latent grid")

        def decode_pressure_chunk(index):
            gather = getattr(self, f"gathers{index}")
            subgather = getattr(self, f"subgathers{index}")
            edge = getattr(self, f"edge_features{index}")
            deterministic = []
            quantiles = []
            for level_group in range(5):
                values = x[level_group][:, gather][:, subgather][:, :, :, :, 0, :]
                values = torch.cat(
                    (values, edge[None].expand(batch, -1, -1, -1, -1)),
                    axis=4,
                ).flatten(start_dim=3)
                pre = self.prefinal_pr(values)
                values = self.postfinal_pr(pre + self.final_pr(pre))
                deterministic_size = pressure_variables * self.config.patch_size[0]
                latitudes, longitudes = values.shape[1:3]
                deterministic.append(
                    values[..., :deterministic_size].view(
                        batch,
                        latitudes,
                        longitudes,
                        pressure_variables,
                        self.config.patch_size[0],
                    )
                )
                quantiles.append(
                    values[..., deterministic_size:].view(
                        batch,
                        latitudes,
                        longitudes,
                        pressure_variables,
                        1,
                        self.n_quantiles,
                    )
                )
            return (
                torch.cat(deterministic, axis=4).flatten(start_dim=3),
                torch.cat(quantiles, axis=4).flatten(start_dim=3),
            )

        surface_only = x[0] is None
        if not surface_only:
            decoded = [decode_pressure_chunk(index) for index in range(self.n_chunks)]
            pressure_deterministic = torch.cat([item[0] for item in decoded], axis=1)
            pressure_quantiles = torch.cat([item[1] for item in decoded], axis=1)

        surface_latent = x[-1]

        def decode_surface_chunk(index):
            gather = getattr(self, f"gathers{index}")
            subgather = getattr(self, f"subgathers{index}")
            edge = getattr(self, f"edge_features{index}")
            values = surface_latent[:, gather][:, subgather][:, :, :, :, 0]
            values = torch.cat(
                (values, edge[None].expand(batch, -1, -1, -1, -1)),
                axis=4,
            ).flatten(start_dim=3)
            pre = self.prefinal_sfc(values)
            return self.postfinal_sfc(pre + self.final_sfc(pre))

        surface = torch.cat(
            [decode_surface_chunk(index) for index in range(self.n_chunks)],
            axis=1,
        )
        if surface_only:
            return surface
        deterministic_count = len(self.sfc_det_vars)
        return torch.cat(
            (
                pressure_deterministic,
                surface[..., :deterministic_count],
                pressure_quantiles,
                surface[..., deterministic_count:],
            ),
            axis=-1,
        )
