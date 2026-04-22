import torch
import torch.nn as nn
from e3nn.o3 import spherical_harmonics

class RPEncoding(nn.Module):
    def __init__(self, num_radial=64, lmax=3, cutoff=10.0):
        super().__init__()
        self.num_radial = num_radial
        self.lmax = lmax
        self.cutoff = cutoff

        centers = torch.linspace(0.01 * cutoff, 0.99 * cutoff, num_radial)
        self.register_buffer("centers", centers)
        self.register_buffer("width", (centers[1] - centers[0]))

        self.sh_dim = sum([2 * l + 1 for l in range(lmax + 1)])
        self.out_dim = self.sh_dim * num_radial

    def forward(self, distances, directions):
        """
        Args:
            distances: [B, L, L] - interatomic distances
            directions: [B, L, L, 3] - unit direction vectors

        Returns:
            [B, L, L, out_dim] 
        """
        B, L, _, _ = directions.shape

        d = distances.unsqueeze(-1)  # [B, L, L, 1]
        rbf = torch.exp(-((d - self.centers) ** 2) / (2 * self.width ** 2))  # [B, L, L, num_radial]

        sph = spherical_harmonics(list(range(self.lmax + 1)), directions, normalize=True, normalization='component')  # [B, L, L, sh_dim]

        rbf = rbf.unsqueeze(-1)      # [B, L, L, num_radial, 1]
        sph = sph.unsqueeze(-2)      # [B, L, L, 1, sh_dim]
        con = rbf * sph              # [B, L, L, num_radial, sh_dim]
        con = con.view(B, L, L, -1)  # [B, L, L, out_dim]

        return con
