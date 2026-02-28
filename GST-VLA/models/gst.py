"""
Gaussian Spatial Tokenizer (GST)
=================================
NOVEL TRAINABLE MODULE — DEAD-VLA, ACCV 2026

Key contribution C1: converts RGB patch features + metric depth
into structured 3D Gaussian tokens z_spatial ∈ R^(B, N_g, d_gst).

Pipeline:
    ① Back-project patch centers to 3D:  p_i = D̂ K^{-1} [u,v,1]^T
    ② Predict Gaussian params:           G = MLP(f_sem) → {μ, Σ, α}
    ③ 3D Fourier positional encoding:    γ(p) = [sin,cos]^L Fourier
    ④ Spatial aggregation:               radius grouping + attn pool
    → z_spatial ∈ R^(B, N_g, d_gst)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from utils.geometry import backproject_to_3d, build_camera_intrinsics
from utils.fourier import FourierPositionalEncoding3D


# ─────────────────────────────────────────────
# Gaussian Parameter Head
# ─────────────────────────────────────────────

class GaussianParamHead(nn.Module):
    """
    MLP that predicts per-point Gaussian parameters from semantic features.

    Input:  f_sem ∈ R^(B, N_patches, d_sem)   [SigLIP features]
    Output:
        mu_offset ∈ R^(B, N, 3)   — Gaussian center offset in 3D
        log_scale ∈ R^(B, N, 3)   — log scale (diagonal Σ)
        alpha     ∈ R^(B, N, 1)   — opacity/confidence ∈ (0,1)
    """

    def __init__(self, d_sem: int = 1152, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_sem, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head_mu    = nn.Linear(hidden_dim, 3)
        self.head_scale = nn.Linear(hidden_dim, 3)
        self.head_alpha = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.head_mu.weight)
        nn.init.zeros_(self.head_mu.bias)
        nn.init.zeros_(self.head_scale.bias)
        nn.init.constant_(self.head_alpha.bias, 0.5)

    def forward(self, f_sem: torch.Tensor):
        h = self.mlp(f_sem)
        mu_offset = self.head_mu(h)                     # (B, N, 3)
        log_scale  = self.head_scale(h)                 # (B, N, 3)
        alpha      = torch.sigmoid(self.head_alpha(h))  # (B, N, 1)
        return mu_offset, log_scale, alpha


# ─────────────────────────────────────────────
# Spatial Attention Pooling (Radius Grouping)
# ─────────────────────────────────────────────

class SpatialAttentionPooling(nn.Module):
    """
    Groups N_patches points into N_g Gaussian tokens via cross-attention.

    Learnable cluster queries attend to weighted point features,
    producing z_spatial ∈ R^(B, N_g, d_gst).
    """

    def __init__(
        self,
        N_g: int = 128,
        d_in: int = 1248,    # d_sem(1152) + d_fourier(96)
        d_out: int = 512,    # d_gst
        n_heads: int = 8,
    ):
        super().__init__()
        self.N_g  = N_g
        self.d_out = d_out

        # Learnable cluster center queries
        self.cluster_queries = nn.Parameter(torch.randn(N_g, d_in) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_in,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm1 = nn.LayerNorm(d_in)
        self.norm2 = nn.LayerNorm(d_in)

        # FFN: d_in → d_out (changes dimension)
        self.ffn = nn.Sequential(
            nn.Linear(d_in, d_in * 2),
            nn.GELU(),
            nn.Linear(d_in * 2, d_out),
        )
        self.norm3 = nn.LayerNorm(d_out)

    def forward(
        self,
        point_features: torch.Tensor,   # (B, N, d_in)
        alpha: torch.Tensor,            # (B, N, 1) opacity weights
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = point_features.shape[0]

        queries = self.cluster_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N_g, d_in)

        # Weight point features by opacity
        weighted = point_features * alpha  # (B, N, d_in)

        # Cross-attention: clusters ← weighted points
        q = self.norm1(queries)
        attn_out, _ = self.cross_attn(
            query=q, key=weighted, value=weighted,
            key_padding_mask=key_padding_mask,
        )
        queries = queries + attn_out
        queries = self.norm2(queries)

        # Project to d_out
        z_spatial = self.ffn(queries)
        z_spatial = self.norm3(z_spatial)
        return z_spatial  # (B, N_g, d_out)


# ─────────────────────────────────────────────
# Full Gaussian Spatial Tokenizer
# ─────────────────────────────────────────────

class GaussianSpatialTokenizer(nn.Module):
    """
    GST: converts (f_sem, depth_map) → z_spatial ∈ R^(B, N_g, d_gst)

    Args:
        d_sem:         SigLIP feature dim (1152 for ViT-SO400M)
        d_gst:         output token dim
        N_g:           number of Gaussian spatial tokens
        fourier_bands: Fourier frequency bands (d_fourier = 3*2*bands = 96)
        img_size:      input image size
        patch_size:    ViT patch size (14 for SO400M)
    """

    def __init__(
        self,
        d_sem: int = 1152,
        d_gst: int = 512,
        N_g: int = 128,
        fourier_bands: int = 16,
        img_size: int = 224,
        patch_size: int = 14,
        attn_heads: int = 8,
    ):
        super().__init__()
        self.d_sem = d_sem
        self.d_gst = d_gst
        self.N_g   = N_g
        self.img_size   = img_size
        self.patch_size = patch_size
        self.n_patches_side = img_size // patch_size  # 16
        self.N_patches = self.n_patches_side ** 2     # 256

        # ① Gaussian parameter prediction head
        self.gaussian_head = GaussianParamHead(d_sem=d_sem, hidden_dim=512)

        # ② 3D Fourier positional encoding
        #    include_raw=False → out_dim = 3 * 2 * fourier_bands = 96
        d_fourier = 3 * 2 * fourier_bands  # 96
        self.pos_enc_3d = FourierPositionalEncoding3D(
            num_bands=fourier_bands, include_raw=False
        )
        assert self.pos_enc_3d.out_dim == d_fourier, (
            f"Fourier PE out_dim mismatch: {self.pos_enc_3d.out_dim} vs {d_fourier}"
        )

        # ③ Spatial attention pooling
        d_in = d_sem + d_fourier  # 1152 + 96 = 1248
        self.attn_pool = SpatialAttentionPooling(
            N_g=N_g, d_in=d_in, d_out=d_gst, n_heads=attn_heads,
        )

        # Learnable depth normalisation parameters
        self.depth_scale  = nn.Parameter(torch.ones(1))
        self.depth_offset = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        f_sem: torch.Tensor,           # (B, N_patches, d_sem)
        depth_map: torch.Tensor,       # (B, H, W) metric depth
        K: Optional[torch.Tensor] = None,  # (B, 3, 3)
        robot_state: Optional[torch.Tensor] = None,  # reserved
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            z_spatial: (B, N_g, d_gst)
            aux:       dict with intermediate outputs
        """
        B = f_sem.shape[0]
        device = f_sem.device
        H, W = depth_map.shape[-2:]

        # ── ① Back-project patch centers ───────────────────
        patch_coords_uv = self._get_patch_centers(B, device)   # (B, N, 2)
        patch_depth = self._sample_depth_at_patches(
            depth_map, patch_coords_uv, H, W
        )  # (B, N, 1)
        patch_depth_norm = patch_depth * self.depth_scale + self.depth_offset

        if K is None:
            K = build_camera_intrinsics(H, W, device=device).unsqueeze(0).expand(B, -1, -1)

        points_3d = backproject_to_3d(patch_coords_uv, patch_depth_norm, K)  # (B, N, 3)

        # ── ② Gaussian parameters ──────────────────────────
        mu_offset, log_scale, alpha = self.gaussian_head(f_sem)
        mu_3d = points_3d + mu_offset  # (B, N, 3)

        # ── ③ 3D Fourier positional encoding ──────────────
        pos_enc = self.pos_enc_3d(mu_3d)  # (B, N, d_fourier=96)

        # ── ④ Aggregate → Gaussian tokens ─────────────────
        point_features = torch.cat([f_sem, pos_enc], dim=-1)  # (B, N, 1248)
        z_spatial = self.attn_pool(point_features, alpha)      # (B, N_g, d_gst)

        aux = {
            "mu_3d":       mu_3d,        # (B, N, 3)
            "log_scale":   log_scale,    # (B, N, 3)
            "alpha":       alpha,        # (B, N, 1)
            "points_3d":   points_3d,    # (B, N, 3)
            "patch_depth": patch_depth,  # (B, N, 1)
        }
        return z_spatial, aux

    def _get_patch_centers(self, B: int, device: torch.device) -> torch.Tensor:
        """Pixel coordinates [u,v] of each ViT patch center. Returns (B, N_patches, 2)."""
        n  = self.n_patches_side
        ps = self.patch_size
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(n, device=device) * ps + ps // 2,  # v (row)
                torch.arange(n, device=device) * ps + ps // 2,  # u (col)
                indexing="ij",
            ),
            dim=-1,
        )  # (n, n, 2) — [v, u]
        coords = coords.reshape(1, -1, 2).expand(B, -1, -1).float()
        return coords[..., [1, 0]]  # reorder to [u, v]

    def _sample_depth_at_patches(
        self,
        depth_map: torch.Tensor,  # (B, H, W)
        uv: torch.Tensor,         # (B, N, 2) pixel coords [u,v]
        H: int, W: int,
    ) -> torch.Tensor:
        """Bilinear sample of depth at patch center pixels. Returns (B, N, 1)."""
        grid = uv.clone()
        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1
        grid = grid.unsqueeze(2)  # (B, N, 1, 2)

        depth_4d = depth_map.unsqueeze(1)  # (B, 1, H, W)
        sampled = F.grid_sample(
            depth_4d, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )  # (B, 1, N, 1)
        return sampled.squeeze(1).squeeze(-1).unsqueeze(-1)  # (B, N, 1)


# ─────────────────────────────────────────────
# GST → VLM Cross-Attention Projector
# ─────────────────────────────────────────────

class GSTtoVLMProjector(nn.Module):
    """
    Trainable bridge: d_gst → d_vlm
    W_proj maps Gaussian tokens to VLM hidden dimension.
    """

    def __init__(self, d_gst: int = 512, d_vlm: int = 3584):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_gst, d_vlm),
            nn.LayerNorm(d_vlm),
        )

    def forward(self, z_spatial: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_spatial: (B, N_g, d_gst)
        Returns:
            spatial_tokens: (B, N_g, d_vlm)
        """
        return self.proj(z_spatial)
