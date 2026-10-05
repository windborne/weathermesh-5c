import torch

from ...assets import RuntimeBundle
from ..meshes import LatLonGrid


class ModelConfig:
    """Architecture and runtime assets for the fixed WeatherMesh-5c model."""

    def __init__(
        self,
        *,
        bundle: RuntimeBundle,
        input_mesh: LatLonGrid,
        output_mesh: LatLonGrid,
        latent_size: int,
        pressure_depth: int,
        encoder_decoder_depth: int,
        patch_size: tuple[int, int, int],
        decoder_chunks: int,
        decoder_latent_size: int,
        decoder_mlp_size: int,
        fork_depth: int,
        noise_projection_scale: int,
        denoising_hours: int,
    ):
        self.bundle = bundle
        self.inputs = [input_mesh]
        self.outputs = [output_mesh]
        self.latent_size = latent_size
        self.pr_depth = pressure_depth
        self.encdec_tr_depth = encoder_decoder_depth
        self.patch_size = patch_size
        self.processor_dts = [3]
        self.dims_per_head = 32
        self.conv_mlp_dim = 512

        self.sparse_mid_latent = 192
        self.dec_sublatent = decoder_latent_size
        self.deconv_mlp_dim = decoder_mlp_size
        self.n_chunks_dec = decoder_chunks

        self.fork_depth = fork_depth
        self.fork_global = True
        self.noise_proj_scale = noise_projection_scale
        self.noise_scale = 1.0
        self.denoising_dt = denoising_hours

        self.num_heads = self.latent_size // self.dims_per_head
        self.latent_levels = output_mesh.levels[:: -self.patch_size[0]][::-1]

    def move_over(self, device):
        device = torch.device(device)
        if self.hmesh.g2m_idx.device != device:
            self.hmesh = self.hmesh.to(device)
            self.hlmesh = self.hlmesh.to(device)
            self.fork_hlmesh = self.fork_hlmesh.to(device)

    def apply_assimilation(self, config):
        for name, value in vars(config).items():
            setattr(self, name, value)
