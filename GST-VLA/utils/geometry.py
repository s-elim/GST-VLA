"""
3D Geometry Utilities for DEAD-VLA
====================================
Back-projection, camera intrinsics, coordinate transforms.
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple


def build_camera_intrinsics(
    H: int,
    W: int,
    fov_deg: float = 60.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build a default pinhole camera intrinsic matrix K (3×3).
    Assumes horizontal FOV = fov_deg degrees.

    Returns:
        K: (3, 3) tensor
    """
    fx = W / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    fy = fx
    cx = W / 2.0
    cy = H / 2.0

    K = torch.tensor([
        [fx,  0., cx],
        [0.,  fy, cy],
        [0.,  0.,  1.],
    ], dtype=torch.float32, device=device)
    return K


def backproject_to_3d(
    uv: torch.Tensor,       # (B, N, 2) pixel coords [u, v]
    depth: torch.Tensor,    # (B, N, 1) or (B, N) metric depth
    K: torch.Tensor,        # (B, 3, 3) or (3, 3) camera intrinsics
) -> torch.Tensor:
    """
    Back-project 2D pixel coordinates + depth → 3D camera-frame points.

    Formula:  p_i = D_i * K^{-1} * [u, v, 1]^T

    Args:
        uv:    (B, N, 2)  pixel coordinates [u=col, v=row]
        depth: (B, N, 1) or (B, N)  metric depth in meters
        K:     (B, 3, 3) or (3, 3) camera intrinsic matrix

    Returns:
        points_3d: (B, N, 3) 3D points in camera frame [X, Y, Z]
    """
    B, N, _ = uv.shape
    device = uv.device

    # Ensure depth is (B, N, 1)
    if depth.dim() == 2:
        depth = depth.unsqueeze(-1)

    # Build homogeneous 2D coords: [u, v, 1]
    ones = torch.ones(B, N, 1, device=device, dtype=uv.dtype)
    uv_h = torch.cat([uv, ones], dim=-1)  # (B, N, 3)

    # Handle K shape
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, -1, -1)  # (B, 3, 3)

    # K_inv: (B, 3, 3)
    K_inv = torch.linalg.inv(K)

    # Unproject: p_normalized = K^{-1} [u, v, 1]^T
    p_norm = torch.bmm(K_inv, uv_h.permute(0, 2, 1))  # (B, 3, N)
    p_norm = p_norm.permute(0, 2, 1)  # (B, N, 3)

    # Scale by depth
    points_3d = p_norm * depth  # (B, N, 3) — [X, Y, Z] in camera frame
    return points_3d


def normalize_points_scene(
    points_3d: torch.Tensor,  # (B, N, 3)
    method: str = "scene_center",
) -> Tuple[torch.Tensor, dict]:
    """
    Normalize 3D points to a canonical scale for stable training.

    Args:
        points_3d: raw 3D points
        method: 'scene_center' | 'unit_sphere' | 'none'

    Returns:
        normalized points, normalization params (for inverse)
    """
    if method == "none":
        return points_3d, {}

    if method == "scene_center":
        center = points_3d.mean(dim=1, keepdim=True)
        scale  = points_3d.std(dim=1, keepdim=True).clamp(min=1e-6)
        return (points_3d - center) / scale, {"center": center, "scale": scale}

    if method == "unit_sphere":
        center = points_3d.mean(dim=1, keepdim=True)
        pts_c  = points_3d - center
        radius = pts_c.norm(dim=-1).max(dim=1, keepdim=True).values.unsqueeze(-1).clamp(min=1e-6)
        return pts_c / radius, {"center": center, "radius": radius}

    raise ValueError(f"Unknown normalization method: {method}")


def compute_affine_invariant_depth(
    depth: torch.Tensor,  # (B, H, W)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Affine-invariant depth normalization (as used by Depth Anything V2).
    Scales depth to [0, 1] per-sample using robust percentiles.

    d_norm = (d - d_2%) / (d_98% - d_2% + eps)
    """
    B = depth.shape[0]
    d_flat = depth.reshape(B, -1)
    d_lo  = torch.quantile(d_flat, 0.02, dim=1).view(B, 1, 1)
    d_hi  = torch.quantile(d_flat, 0.98, dim=1).view(B, 1, 1)
    return ((depth - d_lo) / (d_hi - d_lo + eps)).clamp(0.0, 1.0)
