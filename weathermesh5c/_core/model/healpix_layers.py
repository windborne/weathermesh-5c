from torch import nn


class HealpixUpsample(nn.Module):
    """Upsample each nested HEALPix cell with a learned linear projection."""

    def __init__(
        self,
        input_latent_dim,
        output_latent_dim,
        input_healpix_depth,
        output_healpix_depth,
    ):
        super().__init__()
        self.Ci = input_latent_dim
        self.Co = output_latent_dim
        self.D1 = input_healpix_depth
        self.D2 = output_healpix_depth
        self.U = 4 ** (self.D2 - self.D1)
        self.deconv = nn.Linear(self.Ci, self.U * self.Co)

        def project(x):
            batch, vertices, levels, _ = x.shape
            x = self.deconv(x).view(batch, vertices, levels, self.U, self.Co)
            x = x.permute(0, 1, 3, 2, 4)
            return x.reshape(batch, vertices * self.U, levels, self.Co)

        self.cp1 = project

    def forward(self, x):
        return self.cp1(x)
