"""
Frozen Encoder Wrappers for DEAD-VLA
=====================================
1. SigLIP ViT-SO400M/14  → semantic patch features f_sem ∈ R^(B,256,1152)  [FROZEN]
2. Depth Anything V2 (ViT-L) → metric depth D̂ ∈ R^(B,H,W)               [FROZEN]

Both branches are frozen during all training stages.
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
    Output: patch features ∈ R^(B, 256, 1152)   [256 = (224/14)², d=1152]
    """

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-224"):
        super().__init__()
        self.model_name = model_name
        self._model = None

    def _load(self):
        from transformers import SiglipVisionModel
        import torch
        self._model = SiglipVisionModel.from_pretrained(
            self.model_name,
            use_safetensors=True,
            torch_dtype=torch.bfloat16
        )
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=False)
        hidden = outputs.last_hidden_state  # (B, N, d)
        # Remove CLS token if present
        if hidden.shape[1] == 257:
            hidden = hidden[:, 1:, :]
        return hidden  # (B, 256, 1152)

    def get_d_sem(self) -> int:  return 1152
    def get_n_patches(self) -> int:  return 256


class SigLIPEncoderMock(nn.Module):
    """Mock SigLIP encoder — same output shape, no model loading."""

    def __init__(self, d_sem: int = 1152, n_patches: int = 256):
        super().__init__()
        self.d_sem = d_sem
        self.n_patches = n_patches
        self._dummy = nn.Linear(3, d_sem)  # ensures it has parameters

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
    Frozen Depth Anything V2 (ViT-L) affine-invariant depth estimator.

    Input:  RGB image ∈ R^(B, 3, H, W)
    Output: metric depth D̂ ∈ R^(B, H, W) in meters
    """

    def __init__(
        self,
        model_size: str = "vitl",
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.model_size = model_size
        self.pretrained_path = pretrained_path
        self._model = None

    def _load(self):
        try:
            from depth_anything_v2.dpt import DepthAnythingV2 as DAv2
            model_configs = {
                "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            }
            self._model = DAv2(**model_configs[self.model_size])
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
        if self.model is None:
            B, _, H, W = pixel_values.shape
            return torch.rand(B, H, W, device=pixel_values.device) * 5.0
        depth = self.model(pixel_values)
        if depth.dim() == 4:
            depth = depth.squeeze(1)
        return depth  # (B, H, W)


class DepthAnythingV2Mock(nn.Module):
    """Mock depth encoder for testing."""

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, _, H, W = pixel_values.shape
        return torch.rand(B, H, W, device=pixel_values.device) * 3.0 + 0.5


# ─────────────────────────────────────────────
# Combined Dual Encoder
# ─────────────────────────────────────────────

class DualEncoder(nn.Module):
    """
    Combined FROZEN encoder:
        RGB → SigLIP → f_sem ∈ R^(B, 256, 1152)
        RGB → DepthAnythingV2 → D̂ ∈ R^(B, H, W)
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
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_sem     = self.semantic_enc(rgb)   # (B, 256, 1152)
        depth_map = self.depth_enc(rgb)      # (B, H, W)
        return f_sem, depth_map
