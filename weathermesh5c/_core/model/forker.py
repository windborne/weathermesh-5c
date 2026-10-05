import torch
from torch import nn

from .healpix_transformer import ConditionalHealpixBlock


class EnsembleForker(nn.Module):
    """Create forecast ensemble members from a shared latent state."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.noise_proj_scale = getattr(config, "noise_proj_scale", 4)
        self.noise_proj = nn.Linear(
            config.latent_size // self.noise_proj_scale,
            config.latent_size,
        )
        self.fork_depth = getattr(config, "fork_depth", 2)
        self.denoising_dt = getattr(config, "denoising_dt", 12)
        self.fork_blocks = nn.ModuleList(
            ConditionalHealpixBlock(
                dim=config.latent_size,
                num_heads=config.num_heads,
                hlmesh=config.fork_hlmesh,
            )
            for _ in range(self.fork_depth)
        )

    def forward(self, x, n_fork, fork_hour):
        if fork_hour % self.denoising_dt:
            raise ValueError(
                f"fork_hour ({fork_hour}) must be a multiple of denoising_dt ({self.denoising_dt})"
            )

        output_count = x.shape[0] * n_fork
        noise_width = self.config.latent_size // self.noise_proj_scale
        if self.config.fork_global:
            noise = torch.randn((output_count, 1, 1, noise_width), device=x.device)
            noise = noise * self.config.noise_scale
            noise = self.noise_proj(noise)
            noise = noise.repeat(1, x.shape[1], 1, 1)
        else:
            noise = torch.randn(
                (output_count, x.shape[1], 1, noise_width),
                device=x.device,
            )
            noise = noise * self.config.noise_scale
            noise = self.noise_proj(noise)

        ensemble = x.repeat(n_fork, 1, 1, 1)
        for _ in range(fork_hour // self.denoising_dt):
            for block in self.fork_blocks:
                ensemble = block(ensemble, noise)
        return ensemble
