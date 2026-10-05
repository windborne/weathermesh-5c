import torch
from torch import nn

from ...assets import load_archive
from .data import ADDITIONAL_VARIABLES, get_additional_vars, get_constant_vars
from .healpix_transformer import HealpixProcessor


class GlobalEncoder(nn.Module):
    def __init__(self, mesh, config):
        super().__init__()
        self.mesh = mesh
        self.config = config

        c = config
        data, cvars = get_constant_vars(
            mesh, config.bundle.global_root, resolve=config.bundle.global_path
        )
        data = data.unsqueeze(0)
        self.register_buffer("const_data", data)
        self.total_sfc = self.mesh.n_sfc_vars + ADDITIONAL_VARIABLES + len(cvars)

        depth = 7
        grid = load_archive(config.bundle.global_path(f"meshes/healpix_g2m_{depth}.npz"))
        g2m_idx = grid["g2m_idx"]
        edge_feats = grid["edge_features"]
        self.register_buffer("g2m_idx", torch.from_numpy(g2m_idx))
        self.register_buffer("edge_feats", torch.from_numpy(edge_feats))

        nneigh = self.g2m_idx.shape[1]

        patch = self.config.patch_size
        assert patch[1] == patch[2]

        gelu = nn.GELU(approximate="tanh")

        factor2 = 4 ** (depth - self.config.hlmesh.depth)

        self.preinitial_pr = nn.Linear(
            (self.mesh.n_pr_vars * patch[0] + self.edge_feats.shape[2]) * nneigh, c.conv_mlp_dim
        )
        self.initial_pr = nn.Sequential(gelu, nn.Linear(c.conv_mlp_dim, c.conv_mlp_dim), gelu)
        self.postinitial_pr = nn.Linear(c.conv_mlp_dim, c.latent_size // factor2)

        self.preinitial_sfc = nn.Linear(
            (self.total_sfc + self.edge_feats.shape[2]) * nneigh, c.conv_mlp_dim
        )
        self.initial_sfc = nn.Sequential(gelu, nn.Linear(c.conv_mlp_dim, c.conv_mlp_dim), gelu)
        self.postinitial_sfc = nn.Linear(c.conv_mlp_dim, c.latent_size // factor2)

        self.patch_vertical = patch[0]

        self.factor2 = factor2

        self.tr = HealpixProcessor(
            depth=self.config.encdec_tr_depth,
            num_heads=c.num_heads,
            dim=c.latent_size,
            hlmesh=self.config.hlmesh,
        )

    def update(self, config):
        self.config = config
        self.tr.update(self.config.hlmesh)

    def _encode_chunks(self, x, t0s):
        addl = get_additional_vars(t0s)

        B, H, W, _ = x.shape
        Cpr = self.mesh.n_pr_vars
        D = self.mesh.n_levels
        c = self.config

        x_pr = x[:, :, :, : self.mesh.n_pr].view(B, H * W, Cpr, D)
        nneigh = self.g2m_idx.shape[1]
        nvertical = D // self.patch_vertical

        x_sfc = x[:, :, :, self.mesh.n_pr :]
        surface = torch.cat((x_sfc, addl, self.const_data), dim=-1)
        surface = surface.view(B, H * W, self.total_sfc)

        batch_size = 12288 * 4

        def encode_chunk(start):
            sl = slice(start, start + batch_size)
            x_pr_gath = x_pr[:, self.g2m_idx[sl]]
            Nloc = x_pr_gath.shape[1]
            x_pr_gath = x_pr_gath.view(
                B, Nloc, nneigh, Cpr, nvertical, self.patch_vertical
            )  # eg. [1, 49152, 4, 5, 5, 5]
            x_pr_gath = x_pr_gath.permute(0, 1, 4, 2, 3, 5).flatten(
                start_dim=4
            )  # eg. [1, 49152, 5, 4, 25]
            x_pr_gath = torch.cat(
                (x_pr_gath, self.edge_feats[None, sl, None].expand(-1, -1, nvertical, -1, -1)),
                dim=4,
            )  # eg. [1, 49152, 5, 4, 43]
            pre = self.preinitial_pr(x_pr_gath.flatten(start_dim=3))
            x_pr_gath = self.initial_pr(pre)
            x_pr_gath = pre + x_pr_gath
            x_pr_gath = self.postinitial_pr(x_pr_gath)
            x_pr_gath = x_pr_gath.view(
                B, Nloc // self.factor2, self.factor2, nvertical, c.latent_size // self.factor2
            )
            x_pr_gath = x_pr_gath.permute(0, 1, 3, 2, 4).flatten(start_dim=3)

            x_sfc_gath = surface[:, self.g2m_idx[sl]]
            x_sfc_gath = torch.cat((x_sfc_gath, self.edge_feats[None, sl]), dim=3)

            pre = self.preinitial_sfc(x_sfc_gath.flatten(start_dim=2))
            x_sfc_gath = self.initial_sfc(pre)
            x_sfc_gath = pre + x_sfc_gath
            x_sfc_gath = self.postinitial_sfc(x_sfc_gath)
            x_sfc_gath = x_sfc_gath.view(B, Nloc // self.factor2, c.latent_size)

            x = torch.concat((x_pr_gath, x_sfc_gath[:, :, None, :]), dim=2)
            return x

        return [encode_chunk(start) for start in range(0, self.g2m_idx.shape[0], batch_size)]

    def forward(self, x, t0s):
        assert not torch.isnan(x).any()

        x = torch.cat(self._encode_chunks(x, t0s), dim=1)

        x += self.config.hlmesh.posemb[None]

        x = self.tr(x)

        return x
