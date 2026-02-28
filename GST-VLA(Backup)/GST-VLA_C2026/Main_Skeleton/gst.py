"""
Gaussian Spatial Tokenizer (GST)
=================================
NOVEL TRAINABLE MODULE for GST-VLA (ACCV 2026)

Paper: GST-VLA: Structured Gaussian Spatial Tokens for
       3D-Aware Vision-Language-Action Models

Pipeline:
    1. Back-project patch features to 3D using metric depth
    2. Predict per-point Gaussian parameters (μ, Σ, α) via MLP
    3. Apply 3D Fourier positional encoding to 3D positions
    4. Aggregate via radius grouping + attention pooling
    → z_spatial ∈ R^(N_g × d_gst)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

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
        mu    ∈ R^(B, N_patches, 3)   — Gaussian center offset in 3D
        log_s ∈ R^(B, N_patches, 3)   — log scale (diagonal Σ)
        alpha ∈ R^(B, N_patches, 1)   — opacity/confidence weight
    """

    def __init__(self, d_sem: int = 1152, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_sem, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head_mu    = nn.Linear(hidden_dim, 3)   # center offset
        self.head_scale = nn.Linear(hidden_dim, 3)   # log-scale diagonal
        self.head_alpha = nn.Linear(hidden_dim, 1)   # opacity

        # Init: small offsets, unit scale, mid opacity
        nn.init.zeros_(self.head_mu.weight)
        nn.init.zeros_(self.head_mu.bias)
        nn.init.zeros_(self.head_scale.bias)
        nn.init.constant_(self.head_alpha.bias, 0.5)

    def forward(self, f_sem: torch.Tensor):
        h = self.mlp(f_sem)
        mu_offset = self.head_mu(h)                     # (B, N, 3)
        log_scale  = self.head_scale(h)                 # (B, N, 3) — log σ
        alpha      = torch.sigmoid(self.head_alpha(h))  # (B, N, 1) ∈ (0,1)
        return mu_offset, log_scale, alpha


# ─────────────────────────────────────────────
# Radius Grouping + Attention Pooling
# ─────────────────────────────────────────────

class SpatialAttentionPooling(nn.Module):
    """
    Groups 3D points by N_g learnable cluster centers,
    then performs weighted attention pooling within each group.

    Args:
        N_g:       number of output Gaussian tokens
        d_in:      input feature dim (d_sem + d_pos_enc)
        d_out:     output feature dim (d_gst)
        n_heads:   number of attention heads
    """

    def __init__(
        self,
        N_g: int = 128,
        d_in: int = 1152 + 128,   # d_sem + fourier_dim
        d_out: int = 512,
        n_heads: int = 8,
    ):
        super().__init__()
        self.N_g = N_g
        self.d_out = d_out

        # Learnable cluster center queries (in 3D feature space)
        self.cluster_queries = nn.Parameter(torch.randn(N_g, d_in) * 0.02)

        # Cross-attention: queries=clusters, keys/values=points
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_in,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm1 = nn.LayerNorm(d_in)
        self.norm2 = nn.LayerNorm(d_in)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_in, d_in * 2),
            nn.GELU(),
            nn.Linear(d_in * 2, d_out),
        )
        self.norm3 = nn.LayerNorm(d_out)

    def forward(
        self,
        point_features: torch.Tensor,   # (B, N, d_in)
        alpha: torch.Tensor,            # (B, N, 1)  attention bias
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            z_spatial: (B, N_g, d_out)
        """
        B = point_features.shape[0]

        # Expand cluster queries to batch
        queries = self.cluster_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N_g, d_in)

        # Weight point features by alpha (soft masking)
        weighted_features = point_features * alpha  # (B, N, d_in)

        # Cross-attention: clusters attend to weighted point features
        q = self.norm1(queries)
        attn_out, _ = self.cross_attn(
            query=q,
            key=weighted_features,
            value=weighted_features,
            key_padding_mask=key_padding_mask,
        )
        queries = queries + attn_out                # residual
        queries = self.norm2(queries)

        # FFN projection to d_out
        z_spatial = self.ffn(queries)
        z_spatial = self.norm3(z_spatial)

        return z_spatial   # (B, N_g, d_out)


# ─────────────────────────────────────────────
# Full Gaussian Spatial Tokenizer
# ─────────────────────────────────────────────

class GaussianSpatialTokenizer(nn.Module):
    """
    GST: Gaussian Spatial Tokenizer
    
    Converts (RGB patch features, metric depth map) → structured 3D Gaussian tokens
    
    Args:
        d_sem:         SigLIP feature dim (default 1152 for ViT-SO400M)
        d_gst:         output token dim
        N_g:           number of output Gaussian tokens
        fourier_bands: number of Fourier frequency bands for 3D PE
        img_size:      input image size (H=W assumed)
        patch_size:    ViT patch size
    """

    def __init__(
        self,
        d_sem: int = 1152,
        d_gst: int = 512,
        N_g: int = 128,
        fourier_bands: int = 16,          # → dim = 6*bands = 96... we use 3*2*bands
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
        self.n_patches_side = img_size // patch_size  # 16 for 224/14
        self.N_patches = self.n_patches_side ** 2     # 256

        # ① Gaussian parameter prediction
        self.gaussian_head = GaussianParamHead(d_sem=d_sem, hidden_dim=512)

        # ② 3D Fourier positional encoding
        d_fourier = 3 * 2 * fourier_bands  # sin+cos for x,y,z
        self.pos_enc_3d = FourierPositionalEncoding3D(num_bands=fourier_bands)

        # ③ Spatial attention pooling
        d_in = d_sem + d_fourier
        self.attn_pool = SpatialAttentionPooling(
            N_g=N_g,
            d_in=d_in,
            d_out=d_gst,
            n_heads=attn_heads,
        )

        # Scale/offset learnable for depth normalization
        self.depth_scale  = nn.Parameter(torch.ones(1))
        self.depth_offset = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        f_sem: torch.Tensor,           # (B, N_patches, d_sem) from SigLIP [FROZEN]
        depth_map: torch.Tensor,       # (B, H, W) metric depth in meters
        K: Optional[torch.Tensor] = None,  # (B, 3, 3) camera intrinsics
        robot_state: Optional[torch.Tensor] = None,  # (B, d_state) — future use
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of GST.
        
        Returns:
            z_spatial: (B, N_g, d_gst) — Gaussian spatial tokens
            aux:       dict with intermediate outputs for visualization/loss
        """
        B = f_sem.shape[0]
        device = f_sem.device
        H, W = depth_map.shape[-2:]

        # ── Step 1: Back-project patch centers to 3D ──────────────────
        # Get patch center coordinates in image space
        patch_coords_uv = self._get_patch_centers(B, device)  # (B, N, 2)

        # Sample depth at patch centers (bilinear)
        patch_depth = self._sample_depth_at_patches(
            depth_map, patch_coords_uv, H, W
        )  # (B, N, 1)

        # Normalize depth
        patch_depth_norm = patch_depth * self.depth_scale + self.depth_offset

        # Unproject to 3D using camera intrinsics
        if K is None:
            K = build_camera_intrinsics(H, W, device=device).unsqueeze(0).expand(B, -1, -1)
        
        points_3d = backproject_to_3d(
            patch_coords_uv, patch_depth_norm, K
        )  # (B, N, 3)

        # ── Step 2: Predict Gaussian parameters ───────────────────────
        mu_offset, log_scale, alpha = self.gaussian_head(f_sem)
        # (B, N, 3), (B, N, 3), (B, N, 1)

        # Gaussian centers = backprojected 3D point + learned offset
        mu_3d = points_3d + mu_offset  # (B, N, 3)

        # ── Step 3: 3D Fourier positional encoding ────────────────────
        pos_enc = self.pos_enc_3d(mu_3d)  # (B, N, d_fourier)

        # ── Step 4: Concatenate semantic + positional features ────────
        point_features = torch.cat([f_sem, pos_enc], dim=-1)  # (B, N, d_sem+d_fourier)

        # ── Step 5: Spatial aggregation → Gaussian tokens ─────────────
        z_spatial = self.attn_pool(point_features, alpha)  # (B, N_g, d_gst)

        # Auxiliary outputs
        aux = {
            "mu_3d":      mu_3d,       # (B, N, 3)
            "log_scale":  log_scale,   # (B, N, 3)
            "alpha":      alpha,       # (B, N, 1)
            "points_3d":  points_3d,   # (B, N, 3) backprojected
            "patch_depth": patch_depth, # (B, N, 1)
        }

        return z_spatial, aux

    def _get_patch_centers(self, B: int, device: torch.device) -> torch.Tensor:
        """
        Compute pixel coordinates of ViT patch centers.
        Returns: (B, N_patches, 2) in pixel space [u, v]
        """
        n = self.n_patches_side
        ps = self.patch_size
        # Center of each patch: offset by half patch size
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(n, device=device) * ps + ps // 2,  # v (row)
                torch.arange(n, device=device) * ps + ps // 2,  # u (col)
                indexing="ij",
            ),
            dim=-1,
        )  # (n, n, 2) → [v, u]
        coords = coords.reshape(1, -1, 2).expand(B, -1, -1).float()
        # Reorder to [u, v]
        coords = coords[..., [1, 0]]
        return coords  # (B, N_patches, 2) — u=col, v=row

    def _sample_depth_at_patches(
        self,
        depth_map: torch.Tensor,   # (B, H, W)
        uv: torch.Tensor,          # (B, N, 2) pixel coords [u,v]
        H: int, W: int,
    ) -> torch.Tensor:
        """Bilinear sample depth at patch center pixel coordinates."""
        # Normalize to [-1, 1] for grid_sample
        grid = uv.clone()
        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1  # u → x
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1  # v → y
        grid = grid.unsqueeze(2)  # (B, N, 1, 2)

        depth_4d = depth_map.unsqueeze(1)  # (B, 1, H, W)
        sampled = F.grid_sample(
            depth_4d, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )  # (B, 1, N, 1)
        return sampled.squeeze(1).squeeze(-1).unsqueeze(-1)  # (B, N, 1)


# ─────────────────────────────────────────────
# VLM Cross-Attention Projector
# ─────────────────────────────────────────────

class GSTtoVLMProjector(nn.Module):
    """
    Projects GST tokens (d_gst) to VLM hidden dim (d_vlm).
    Trainable bridge between frozen GST output and frozen VLM.
    
    W_proj: d_gst → d_vlm  (TRAINABLE)
    """

    def __init__(self, d_gst: int = 512, d_vlm: int = 3584):
        """d_vlm=3584 for Qwen2.5-VL-7B"""
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
            spatial_tokens: (B, N_g, d_vlm) — prepended to VLM input
        """
        return self.proj(z_spatial)
