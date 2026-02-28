"""
Frozen Encoder Wrappers for GST-VLA
=====================================
Wraps:
  1. SigLIP ViT-SO400M/14  → semantic patch features (FROZEN)
  2. Depth Anything V2 (ViT-L) → metric depth maps (FROZEN)

Both are kept frozen during all training stages.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# ─────────────────────────────────────────────
# SigLIP Semantic Encoder (FROZEN)
# ─────────────────────────────────────────────

class SigLIPEncoder(nn.Module):
    """
    Frozen SigLIP ViT-SO400M/14 semantic encoder.
    
    Input:  RGB image ∈ R^(B, 3, 224, 224)
    Output: patch features ∈ R^(B, 256, 1152)
            [256 patches = (224/14)^2, d=1152]

    Also outputs language token features when text is provided.
    """

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-224"):
        super().__init__()
        self.model_name = model_name
        self._model = None  # Lazy load

    def _load(self):
        """Lazy load to avoid import overhead when not needed."""
        from transformers import SiglipVisionModel, SiglipProcessor
        self._model = SiglipVisionModel.from_pretrained(self.model_name)
        self._model.eval()
        # Freeze all parameters
        for p in self._model.parameters():
            p.requires_grad_(False)

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,  # (B, 3, 224, 224)
    ) -> torch.Tensor:
        """
        Returns:
            patch_features: (B, 256, 1152) — all patch tokens (no CLS)
        """
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=False)
        # last_hidden_state: (B, 256+1, 1152) — includes CLS token in some configs
        hidden = outputs.last_hidden_state  # (B, N, d)
        
        # Remove CLS token if present (check shape)
        if hidden.shape[1] == 257:
            hidden = hidden[:, 1:, :]   # remove CLS → (B, 256, 1152)
        
        return hidden   # (B, 256, 1152)

    def get_d_sem(self) -> int:
        return 1152

    def get_n_patches(self) -> int:
        return 256


class SigLIPEncoderMock(nn.Module):
    """
    Mock SigLIP encoder for development/testing without loading 400M model.
    Same output shape as real SigLIPEncoder.
    """

    def __init__(self, d_sem: int = 1152, n_patches: int = 256):
        super().__init__()
        self.d_sem = d_sem
        self.n_patches = n_patches
        # Pretend conv to map image → features
        self._dummy = nn.Linear(3 * 14 * 14, d_sem)  # not used in forward

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0]
        return torch.randn(B, self.n_patches, self.d_sem, device=pixel_values.device)

    def get_d_sem(self): return self.d_sem
    def get_n_patches(self): return self.n_patches


# ─────────────────────────────────────────────
# Depth Anything V2 (FROZEN)
# ─────────────────────────────────────────────

class DepthAnythingV2Encoder(nn.Module):
    """
    Frozen Depth Anything V2 (ViT-L) metric depth estimator.
    
    Input:  RGB image ∈ R^(B, 3, H, W)
    Output: metric depth ∈ R^(B, H, W)  in meters
    
    Uses affine-invariant estimation, so outputs are up to scale/shift.
    Absolute metric scale requires additional calibration or multi-frame.
    """

    def __init__(
        self,
        model_size: str = "vitl",   # 'vits', 'vitb', 'vitl'
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.model_size = model_size
        self.pretrained_path = pretrained_path
        self._model = None

    def _load(self):
        """Lazy load Depth Anything V2."""
        try:
            # Official DepthAnythingV2 repo interface
            from depth_anything_v2.dpt import DepthAnythingV2 as DAv2
            model_configs = {
                "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            }
            cfg = model_configs[self.model_size]
            self._model = DAv2(**cfg)
            if self.pretrained_path:
                state = torch.load(self.pretrained_path, map_location="cpu", weights_only=True)
                self._model.load_state_dict(state)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        except ImportError:
            print("[DepthAnythingV2] Package not found, using mock depth encoder.")
            self._model = None

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) normalized RGB

        Returns:
            depth: (B, H, W) metric depth in meters
        """
        if self.model is None:
            # Mock: return random depth
            B, _, H, W = pixel_values.shape
            return torch.rand(B, H, W, device=pixel_values.device) * 5.0

        depth = self.model(pixel_values)
        if depth.dim() == 4:
            depth = depth.squeeze(1)  # (B, 1, H, W) → (B, H, W)
        return depth  # (B, H, W)


class DepthAnythingV2Mock(nn.Module):
    """Mock depth encoder for testing."""

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, _, H, W = pixel_values.shape
        # Simulate plausible depth: center closer, edges farther
        depth = torch.rand(B, H, W, device=pixel_values.device) * 3.0 + 0.5
        return depth


# ─────────────────────────────────────────────
# Combined Encoder Module
# ─────────────────────────────────────────────

class DualEncoder(nn.Module):
    """
    Combined RGB semantic + depth encoder module.
    Both branches are FROZEN.
    
    Returns:
        f_sem:     (B, 256, 1152)  semantic patch features
        depth_map: (B, H, W)       metric depth map
    """

    def __init__(
        self,
        use_mock: bool = False,
        depth_model_size: str = "vitl",
        depth_pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        if use_mock:
            self.semantic_enc = SigLIPEncoderMock()
            self.depth_enc    = DepthAnythingV2Mock()
        else:
            self.semantic_enc = SigLIPEncoder()
            self.depth_enc    = DepthAnythingV2Encoder(
                model_size=depth_model_size,
                pretrained_path=depth_pretrained_path,
            )

        # Freeze everything
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        rgb: torch.Tensor,  # (B, 3, 224, 224)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            f_sem:     (B, 256, 1152)
            depth_map: (B, 224, 224)
        """
        f_sem     = self.semantic_enc(rgb)   # (B, 256, 1152)
        depth_map = self.depth_enc(rgb)      # (B, H, W)
        return f_sem, depth_map
