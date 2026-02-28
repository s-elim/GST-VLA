"""
3D Fourier Positional Encoding for DEAD-VLA
=============================================
γ(p) = [sin(2^0 π p), cos(2^0 π p), ..., sin(2^{L-1} π p), cos(2^{L-1} π p)]
for each of x, y, z coordinates.

Output dim (include_raw=False) = 3 × 2 × num_bands
Output dim (include_raw=True)  = 3 × 2 × num_bands + 3
"""

import torch
import torch.nn as nn
import math


class FourierPositionalEncoding3D(nn.Module):
    """
    NeRF-style Fourier positional encoding for 3D coordinates.

    For each 3D point p = [x, y, z]:
        γ(p) = [sin(2^0 π x), cos(2^0 π x), ..., sin(2^{L-1} π x), cos(2^{L-1} π x),
                sin(2^0 π y), ...   (same for y)
                sin(2^0 π z), ...]  (same for z)

    Args:
        num_bands:    L, number of frequency bands
        include_raw:  whether to include raw xyz coordinates in output
        scale:        input coordinate scale (normalizes coords before encoding)
    """

    def __init__(
        self,
        num_bands: int = 16,
        include_raw: bool = False,   # default False to match d_fourier = 3*2*num_bands
        scale: float = 1.0,
    ):
        super().__init__()
        self.num_bands   = num_bands
        self.include_raw = include_raw
        self.scale       = scale

        freqs = 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

        self.out_dim = 3 * 2 * num_bands
        if include_raw:
            self.out_dim += 3

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) 3D coordinates

        Returns:
            enc: (B, N, out_dim)
        """
        B, N, _ = points.shape
        p = points * self.scale

        p_exp = p.unsqueeze(-1)                           # (B, N, 3, 1)
        freqs = self.freqs.view(1, 1, 1, -1)              # (1, 1, 1, L)
        args  = math.pi * p_exp * freqs                   # (B, N, 3, L)

        enc_sin = torch.sin(args)
        enc_cos = torch.cos(args)

        enc = torch.stack([enc_sin, enc_cos], dim=-1)     # (B, N, 3, L, 2)
        enc = enc.reshape(B, N, 3 * 2 * self.num_bands)   # (B, N, 3*2*L)

        if self.include_raw:
            enc = torch.cat([p, enc], dim=-1)

        return enc


class LearnableFourierPositionalEncoding3D(nn.Module):
    """
    Learnable variant: applies Fourier encoding then an MLP projection.
    Allows the model to learn optimal frequency weighting.
    """

    def __init__(self, num_bands: int = 16, d_out: int = 128):
        super().__init__()
        self.fourier = FourierPositionalEncoding3D(num_bands=num_bands, include_raw=True)
        d_fourier = self.fourier.out_dim
        self.proj = nn.Sequential(
            nn.Linear(d_fourier, d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
        )
        self.out_dim = d_out

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3)
        Returns:
            enc: (B, N, d_out)
        """
        return self.proj(self.fourier(points))
