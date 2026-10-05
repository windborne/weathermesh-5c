import torch


def sincos_spatial_embed(thresholds: torch.Tensor, embed_dim: int = 32) -> torch.Tensor:
    """Encode normalized spatial coordinates with sinusoidal features."""
    if not torch.all((thresholds >= 0) & (thresholds <= 1)):
        raise ValueError("Spatial coordinates must be between zero and one")
    if thresholds.dtype != torch.float32:
        raise TypeError("Spatial coordinates must use float32 precision")

    scale = embed_dim**0.25
    periods = 2 * scale**thresholds
    frequencies = torch.arange(
        embed_dim // 2,
        device=thresholds.device,
        dtype=thresholds.dtype,
    )
    angles = 2 * torch.pi / periods[:, :, None] * frequencies[None, None, :]
    return torch.concat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(start_dim=1)
