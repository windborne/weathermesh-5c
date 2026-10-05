import torch
from torch import nn

from .heal import IcoAttention, inference_compile


class ResidualMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


@inference_compile
class HealpixBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        hlmesh=None,
        hidden_dim=None,
        linear_warmup=False,
        linear_forward=False,
        full_skip=False,
        full_skip_forward=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = norm_layer(dim)
        self.attn = IcoAttention(
            dim, num_heads, qkv_bias=qkv_bias, hlmesh=hlmesh, hidden_dim=hidden_dim
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ResidualMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def update_hlmesh(self, hlmesh):
        self.attn.update_hlmesh(hlmesh)

    def forward(self, x):
        shortcut = x
        x = self.attn(self.norm1(x))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class HealpixProcessor(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        hlmesh,
    ):

        super().__init__()
        self.dim = dim
        self.depth = depth

        mlist = []
        for _ in range(depth):
            tb_block = HealpixBlock(dim, num_heads, hlmesh=hlmesh)
            mlist.append(tb_block)
        self.blocks = nn.ModuleList(mlist)

    def update(self, hlmesh=None):
        if hlmesh is not None:
            for blk in self.blocks:
                blk.update_hlmesh(hlmesh)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConditionalHealpixBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        hlmesh=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = ConditionalLayerNorm(dim)
        self.attn = IcoAttention(dim, num_heads, qkv_bias=qkv_bias, hlmesh=hlmesh)

        self.norm2 = ConditionalLayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ResidualMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def update_hlmesh(self, hlmesh):
        self.attn.update_hlmesh(hlmesh)

    def forward(self, x, noise):
        shortcut = x
        x = self.attn(self.norm1(x, noise))
        x = shortcut + x
        return x + self.mlp(self.norm2(x, noise))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

        self.noise_proj = nn.Linear(dim, dim * 2)  # Projects noise to scale and bias

    def forward(self, x, noise):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)

        x_out = self.weight * x_norm + self.bias

        noise_cond = noise.mean(dim=(1, 2))  # [B, C] - average across spatial dims
        cond_params = self.noise_proj(noise_cond)  # [B, C*2]
        cond_scale, cond_bias = cond_params.chunk(2, dim=-1)  # [B, C] each

        cond_scale = cond_scale.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
        cond_bias = cond_bias.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]

        x_out = x_out * (1 + cond_scale) + cond_bias

        return x_out
