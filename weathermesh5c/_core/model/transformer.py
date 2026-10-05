import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class WideSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dim_head: int = 64,
    ):
        super().__init__()
        assert hidden_dim % dim_head == 0, (
            f"hidden_dim must be divisible by dim_head, got {hidden_dim} and {dim_head}"
        )
        self.dim_head = dim_head
        self.num_heads = hidden_dim // dim_head

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=True)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).view(B, N, 3, self.num_heads, self.dim_head)
        del x
        q, k, v = qkv.permute(0, 2, 3, 1, 4).unbind(dim=1)
        del qkv

        assert q.dtype == torch.float16, f"q.dtype is {q.dtype}"
        assert k.dtype == torch.float16, f"k.dtype is {k.dtype}"
        assert v.dtype == torch.float16, f"v.dtype is {v.dtype}"
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            )
        del q, k, v
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        return self.to_out(out)


class WideTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        attention_hidden_dim: int,
        feedforward_hidden_dim: int | None = None,
        dim_head: int = 64,
    ):
        super().__init__()
        feedforward_hidden_dim = feedforward_hidden_dim or dim * 4
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, WideSelfAttention(dim, attention_hidden_dim, dim_head)),
                        PreNorm(dim, FeedForward(dim, feedforward_hidden_dim)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention, feedforward in self.layers:
            x = attention(x) + x
            x = feedforward(x) + x
        return x
