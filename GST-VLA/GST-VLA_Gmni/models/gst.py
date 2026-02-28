import torch
import torch.nn as nn
import math
from typing import Tuple

class FourierPositionEncoding3D(nn.Module):
    """
    Computes Fourier features for 3D points to capture high-frequency spatial details.
    """
    def __init__(self, input_dim: int = 3, num_frequencies: int = 64, scale: float = 10.0):
        super().__init__()
        self.scale = scale
        # Random Fourier features (frozen during training)
        self.register_buffer(
            'B', torch.randn(input_dim, num_frequencies) * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 3D coordinates of shape (Batch, Num_Points, 3)
        Returns:
            Fourier features of shape (Batch, Num_Points, num_frequencies * 2)
        """
        # x @ B -> (Batch, Num_Points, num_frequencies)
        x_proj = (2.0 * math.pi * x) @ self.B
        # Concat sin and cos
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out

class GaussianSpatialTokenizer(nn.Module):
    def __init__(
        self, 
        semantic_dim: int = 1152,  # SigLIP So400M/14 output dim
        embed_dim: int = 3584,     # Qwen2.5-VL 7B hidden dim
        num_out_tokens: int = 128, # GST target sequence length (N_3D)
        fourier_freqs: int = 128   # Defines the resolution of the 3D PE
    ):
        super().__init__()
        self.num_out_tokens = num_out_tokens
        self.embed_dim = embed_dim
        
        # 1. 3D Positional Encoding
        self.pe_3d = FourierPositionEncoding3D(
            input_dim=3, 
            num_frequencies=fourier_freqs
        )
        pe_dim = fourier_freqs * 2
        
        # 2. Gaussian Parameter MLP
        # Fuses 2D semantics + 3D Coordinates + 3D PE into the VLM embedding space
        self.feature_fusion = nn.Sequential(
            nn.Linear(semantic_dim + 3 + pe_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 3. Spatial Aggregation
        # Using a learned query-based cross-attention mechanism for the "radius grouping" 
        # This is more robust than simple MaxPool and acts as a Perceiver Resampler,
        # actively querying the scene for actionable geometry.
        self.token_queries = nn.Parameter(torch.randn(1, num_out_tokens, embed_dim))
        self.aggregation_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=16, 
            batch_first=True
        )
        self.norm_pre = nn.LayerNorm(embed_dim)
        self.norm_post = nn.LayerNorm(embed_dim)

    def backproject_to_3d(self, depth_map: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Back-projects a depth map into 3D point clouds using camera intrinsics.
        Args:
            depth_map: (B, 1, H, W) metric depth from Depth Anything V2
            intrinsics: (B, 3, 3) camera intrinsic matrices
        Returns:
            points_3d: (B, H*W, 3) 3D coordinates
        """
        B, _, H, W = depth_map.shape
        device = depth_map.device

        # Create meshgrid for pixel coordinates (u, v)
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device), 
            indexing='ij'
        )
        
        # (B, H, W, 3)
        uv1 = torch.stack([u_coords, v_coords, torch.ones_like(u_coords)], dim=-1).float()
        uv1 = uv1.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Invert intrinsics (B, 3, 3)
        K_inv = torch.inverse(intrinsics)
        
        # Apply inverse intrinsics: K^-1 @ [u, v, 1]^T
        rays = torch.einsum('bij,bhwj->bhwi', K_inv, uv1)
        
        # Multiply by metric depth to get (X, Y, Z)
        points_3d = rays * depth_map.permute(0, 2, 3, 1) 
        
        return points_3d.view(B, -1, 3) # Flatten spatial dimensions

    def forward(
        self, 
        semantic_features: torch.Tensor, 
        depth_map: torch.Tensor, 
        intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            semantic_features: (B, N_patches, 1152) from frozen SigLIP
            depth_map: (B, 1, H, W) from frozen Depth Anything V2
            intrinsics: (B, 3, 3) camera intrinsics
        Returns:
            z_spatial: (B, 128, 3584) spatial tokens ready for Qwen2.5-VL
        """
        B = semantic_features.shape[0]
        
        # 1. Lift to 3D
        points_3d = self.backproject_to_3d(depth_map, intrinsics) # (B, N_patches, 3)
        
        # 2. Compute 3D Positional Encoding
        p_pe = self.pe_3d(points_3d) # (B, N_patches, pe_dim)
        
        # 3. Fuse Modalities
        # Concatenate Semantics, raw (X,Y,Z), and high-freq 3D PE
        fused_input = torch.cat([semantic_features, points_3d, p_pe], dim=-1)
        
        # Project to VLM embedding dimension
        x = self.feature_fusion(fused_input) # (B, N_patches, 3584)
        
        # 4. Spatial Aggregation (Token Compression)
        # Compress N_patches (e.g., 256) down to N_3D (128) using cross-attention
        queries = self.token_queries.expand(B, -1, -1) # (B, 128, 3584)
        x_norm = self.norm_pre(x)
        
        z_spatial, _ = self.aggregation_attn(
            query=queries,
            key=x_norm,
            value=x_norm
        )
        
        # Residual connection and final norm
        z_spatial = self.norm_post(queries + z_spatial) # (B, 128, 3584)
        
        return z_spatial