import os

import torch
import torch.nn.functional as F
from torch import nn

from ...assets import RuntimeBundle, load_archive
from ..triconatten import tri_conatten
from .icoda import make_da_hgrid


def inference_compile(module):
    if "WB_INFERENCE" not in os.environ:
        return module
    return torch.compile(
        module,
        mode="max-autotune-no-cudagraphs",
        fullgraph=False,
        dynamic=False,
        backend="inductor",
    )


def make_hgrid(bundle, nu, levels, nearest_horizontal, nearest_vertical, batch):
    """Load a precomputed HEALPix attention grid."""
    filename = f"hgrid_nu{nu}_D{levels}_KL{nearest_horizontal}_KD{nearest_vertical}_B{batch}.npz"
    path = bundle.global_path(f"meshes/{filename}")
    if not path.is_file():
        raise FileNotFoundError(f"Required HEALPix attention grid is missing: {path}")
    return load_archive(path)


class LatentMesh:
    def __init__(
        self,
        depth=5,
        D=6,
        KL=18,
        KD=5,
        dim=1024,
        B=8,
        BSh=8,
        bundle: RuntimeBundle | None = None,
        latlon_range=None,
        tri=False,
    ):
        if latlon_range is not None:
            raise ValueError("This inference package supports global HEALPix grids only")

        self.depth = depth
        self.NSIDE = 2**depth
        if isinstance(D, list):
            self.latent_structure = D
            D = sum(len(levels) for levels in D)
        else:
            self.latent_structure = None
        self.D = D
        self.KL = KL
        self.KD = KD
        self.B = B
        self.dim = dim
        self.BS = B
        self.BSh = BSh
        self.latlon_range = None
        self.tri = tri
        if bundle is None:
            raise ValueError("LatentMesh needs a runtime bundle")
        self.bundle = bundle

        if self.latent_structure is None:
            self.hgrid = make_hgrid(bundle, self.NSIDE, self.D, self.KL, self.KD, self.B)
        else:
            self.hgrid = make_da_hgrid(
                bundle,
                self.NSIDE,
                self.latent_structure,
                self.KL,
                self.KD,
                self.B,
            )
        self.vertices = torch.from_numpy(self.hgrid["vertices"])
        self.Nvert = self.vertices.shape[0]
        self.which = torch.from_numpy(self.hgrid["which"])
        mask_name = "mask2" if tri else "mask"
        self.mask = torch.from_numpy(self.hgrid[mask_name])
        self.lonlat = torch.from_numpy(self.hgrid["lonlat"])
        # AIDA adds positions per level group, not over its combined attention grid.
        if self.latent_structure is None:
            self.posemb = self.posemb_mesh(self.lonlat, D, dim, dtype=torch.float16)

    def posemb_mesh(self, lonlat, D, dim, T1=300, T2=10, dtype=torch.float32):
        fourier_dim = dim // 8
        omegas = torch.arange(fourier_dim) / (fourier_dim - 1)
        omega = 1.0 / (T1**omegas)
        omega2 = 1.0 / (T2**omegas)

        lon = torch.deg2rad(lonlat[:, 0])
        lat = torch.deg2rad(lonlat[:, 1])
        scale = 40
        x = torch.cos(lat) * torch.cos(lon) * scale
        y = torch.cos(lat) * torch.sin(lon) * scale
        z = torch.sin(lat) * scale

        x_emb = x[:, None] * omega[None, :]
        y_emb = y[:, None] * omega[None, :]
        z_emb = z[:, None] * omega[None, :]
        levels = torch.arange(D)
        level_emb = levels[:, None] * omega2[None, :]
        horizontal = torch.cat(
            (
                torch.sin(x_emb),
                torch.cos(x_emb),
                torch.sin(y_emb),
                torch.cos(y_emb),
                torch.sin(z_emb),
                torch.cos(z_emb),
            ),
            dim=-1,
        ).to(dtype)
        vertical = torch.cat((torch.sin(level_emb), torch.cos(level_emb)), dim=-1).to(dtype)
        embedding = torch.cat(
            (horizontal[:, None].expand(-1, D, -1), vertical[None].expand(len(lonlat), -1, -1)),
            dim=-1,
        )
        padding = dim - fourier_dim * 8
        return F.pad(embedding, (0, padding)) if padding else embedding

    def to(self, *args, **kwargs):
        if hasattr(self, "vertices"):
            self.vertices = self.vertices.to(*args, **kwargs)
        self.which = self.which.to(*args, **kwargs)
        self.mask = self.mask.to(*args, **kwargs)
        if hasattr(self, "posemb"):
            self.posemb = self.posemb.to(*args, **kwargs)
        return self


class EncoderMesh:
    def __init__(self, depth, bundle: RuntimeBundle, latlon_range=None):
        if latlon_range is not None:
            raise ValueError("This inference package supports global HEALPix grids only")
        self.NSIDE = 2**depth
        self.latlon_range = None
        path = bundle.global_path(f"meshes/healpix_{depth}.npz")
        if not path.is_file():
            raise FileNotFoundError(f"Required encoder mesh is missing: {path}")
        grid = load_archive(path)
        self.vertices = grid["vertices"]
        g2m_idx = grid["g2m_idx"]
        g2m_wts = grid["g2m_weights"]
        self.g2m_wts = torch.from_numpy(g2m_wts).half()
        self.g2m_idx = torch.from_numpy(g2m_idx)

    def to(self, *args, **kwargs):
        self.g2m_wts = self.g2m_wts.to(*args, **kwargs)
        self.g2m_idx = self.g2m_idx.to(*args, **kwargs)
        return self


class IcoAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        hlmesh=None,
        hidden_dim=None,
    ):
        super().__init__()
        self.Nvert = hlmesh.Nvert if hlmesh is not None else None
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, self.hidden_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(self.hidden_dim, dim)
        self.hlmesh = hlmesh

    def update_hlmesh(self, hlmesh):
        self.hlmesh = hlmesh

    def forward(self, x):
        batch = x.shape[0]
        self.Nvert = self.hlmesh.Nvert
        if not self.hlmesh.tri:
            raise RuntimeError("WeatherMesh-5c requires the Triton attention grid")
        if x.shape[1] != self.Nvert or x.shape[2] != self.hlmesh.D:
            raise ValueError(
                f"Expected (*, {self.Nvert}, {self.hlmesh.D}, *), got {tuple(x.shape)}"
            )

        qkv = self.qkv(x).reshape(
            batch,
            self.Nvert,
            self.hlmesh.D,
            3,
            self.num_heads,
            self.head_dim,
        )
        q = qkv[:, :, :, 0]
        q /= self.scale
        k = qkv[:, :, :, 1]
        v = qkv[:, :, :, 2]
        output = [
            tri_conatten(
                q[index],
                k[index],
                v[index],
                self.hlmesh.which,
                self.hlmesh.mask,
                self.hlmesh.BS,
            )
            for index in range(batch)
        ]
        output = output[0][None] if len(output) == 1 else torch.stack(output)
        return self.proj(output)
