"""
GST-VLA: Full Model Assembly
==============================
Paper: GST-VLA: Structured Gaussian Spatial Tokens for
       3D-Aware Vision-Language-Action Models (ACCV 2026)

Architecture:
    RGB + Language + Robot State
        │
        ├─[FROZEN]─► SigLIP ViT-SO400M/14 ──► f_sem ∈ R^(B,256,1152)
        │
        ├─[FROZEN]─► Depth Anything V2 ──────► depth ∈ R^(B,H,W)
        │
        │
        │         [TRAINABLE - NOVEL]
        ├─────────► GST (Gaussian Spatial Tokenizer)
        │              ① Backproject + Gaussian params
        │              ② 3D Fourier PE
        │              ③ Spatial attention pooling
        │           → z_spatial ∈ R^(B, N_g=128, d_gst=512)
        │
        │         [TRAINABLE]
        ├─────────► W_proj: d_gst → d_vlm
        │           → spatial_tokens ∈ R^(B, 128, 3584)
        │
        │         [FROZEN]
        ├─────────► Qwen2.5-VL-7B
        │           [spatial_tokens | language_tokens] → h_vlm
        │           → h_vlm ∈ R^(B, N, d_vlm)
        │
        │         [TRAINABLE]
        └─────────► Flow Matching Action Expert
                    → a_t ∈ R^(B, 16, 7)

TRAINABLE parameters (~45M for stage 1, ~345M total):
  - GST:              ~15M
  - W_proj:           ~2M
  - Flow Expert:      ~300M
  (VLM LoRA added in Stage 2/3)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List

from models.encoders import DualEncoder
from models.gst import GaussianSpatialTokenizer, GSTtoVLMProjector
from models.vlm_wrapper import QwenVLMWrapper
from models.flow_matching import FlowMatchingActionExpert


class GSTVLA(nn.Module):
    """
    GST-VLA: Full model for ACCV 2026 submission.
    
    Trainable components:
      - GST (Gaussian Spatial Tokenizer)
      - GSTtoVLMProjector (W_proj)
      - FlowMatchingActionExpert
    
    Frozen components:
      - SigLIP ViT-SO400M/14
      - Depth Anything V2 (ViT-L)
      - Qwen2.5-VL-7B
    """

    def __init__(
        self,
        # GST config
        d_sem: int = 1152,
        d_gst: int = 512,
        N_g: int = 128,
        fourier_bands: int = 16,
        img_size: int = 224,
        patch_size: int = 14,
        # VLM config
        d_vlm: int = 3584,
        vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        # Action expert config
        d_state: int = 14,
        H: int = 16,
        d_action: int = 7,
        n_expert_layers: int = 8,
        n_euler_steps: int = 10,
        # Runtime config
        use_mock_encoders: bool = False,
        use_mock_vlm: bool = False,
    ):
        super().__init__()

        self.H = H
        self.d_action = d_action

        # ── Frozen Encoders ──────────────────────────────
        self.dual_encoder = DualEncoder(
            use_mock=use_mock_encoders,
        )
        # Ensure frozen
        for p in self.dual_encoder.parameters():
            p.requires_grad_(False)

        # ── Novel Trainable: GST ─────────────────────────
        self.gst = GaussianSpatialTokenizer(
            d_sem=d_sem,
            d_gst=d_gst,
            N_g=N_g,
            fourier_bands=fourier_bands,
            img_size=img_size,
            patch_size=patch_size,
        )

        # ── Trainable: VLM Projector ─────────────────────
        self.vlm_projector = GSTtoVLMProjector(d_gst=d_gst, d_vlm=d_vlm)

        # ── Frozen VLM ───────────────────────────────────
        self.vlm = QwenVLMWrapper(
            model_name=vlm_model_name,
            use_mock=use_mock_vlm,
        )

        # ── Trainable: Flow Matching Action Expert ────────
        self.action_expert = FlowMatchingActionExpert(
            d_model=512,
            d_vlm=d_vlm,
            d_state=d_state,
            H=H,
            d_action=d_action,
            n_layers=n_expert_layers,
            n_euler=n_euler_steps,
        )

    # ─────────────────────────────────────────────────────────
    # Forward (Training)
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        rgb: torch.Tensor,              # (B, 3, 224, 224)
        instruction_ids: torch.Tensor,  # (B, L) tokenized
        attention_mask: torch.Tensor,   # (B, L)
        robot_state: torch.Tensor,      # (B, d_state)
        gt_actions: torch.Tensor,       # (B, H, d_action) for training
        camera_K: Optional[torch.Tensor] = None,  # (B, 3, 3)
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training.
        
        Returns dict with:
            loss:        total training loss
            loss_flow:   flow matching loss
            aux:         auxiliary outputs (3D points, gaussians, etc.)
        """
        # ── Stage 1: Encode ───────────────────────────────
        with torch.no_grad():
            f_sem, depth_map = self.dual_encoder(rgb)  # frozen

        # ── Stage 2: GST → Spatial Tokens ─────────────────
        z_spatial, gst_aux = self.gst(
            f_sem=f_sem,
            depth_map=depth_map,
            K=camera_K,
        )  # (B, N_g, d_gst)

        # ── Stage 3: Project to VLM space ─────────────────
        spatial_tokens_vlm = self.vlm_projector(z_spatial)  # (B, N_g, d_vlm)

        # ── Stage 4: VLM Reasoning (frozen) ───────────────
        with torch.no_grad():
            h_vlm = self.vlm(
                spatial_tokens=spatial_tokens_vlm,
                input_ids=instruction_ids,
                attention_mask=attention_mask,
            )  # (B, N_g+L, d_vlm)

        # ── Stage 5: Flow Matching Loss ────────────────────
        loss_flow = self.action_expert.compute_flow_loss(
            a1=gt_actions,
            h_vlm=h_vlm,
            s_t=robot_state,
        )

        return {
            "loss":      loss_flow,
            "loss_flow": loss_flow,
            "z_spatial": z_spatial,
            "h_vlm":     h_vlm,
            "gst_aux":   gst_aux,
        }

    # ─────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_action(
        self,
        rgb: torch.Tensor,              # (B, 3, 224, 224)
        instruction_ids: torch.Tensor,  # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
        robot_state: torch.Tensor,      # (B, d_state)
        camera_K: Optional[torch.Tensor] = None,
        n_euler_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Inference: predict action chunk a_t ∈ R^(B, H, 7).
        
        Returns:
            actions: (B, H, 7)
            info:    dict with 3D spatial info
        """
        # Encode
        f_sem, depth_map = self.dual_encoder(rgb)

        # GST
        z_spatial, gst_aux = self.gst(f_sem=f_sem, depth_map=depth_map, K=camera_K)

        # Project
        spatial_tokens_vlm = self.vlm_projector(z_spatial)

        # VLM
        h_vlm = self.vlm(
            spatial_tokens=spatial_tokens_vlm,
            input_ids=instruction_ids,
            attention_mask=attention_mask,
        )

        # Sample actions via ODE
        actions = self.action_expert.sample(
            h_vlm=h_vlm,
            s_t=robot_state,
            n_steps=n_euler_steps,
        )  # (B, H, 7)

        info = {
            "z_spatial":  z_spatial,
            "mu_3d":      gst_aux["mu_3d"],
            "alpha":      gst_aux["alpha"],
            "depth_map":  depth_map,
        }

        return actions, info

    # ─────────────────────────────────────────────────────────
    # Parameter counting
    # ─────────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and frozen parameters by component."""

        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        def count_all(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "gst_trainable":         count(self.gst),
            "projector_trainable":   count(self.vlm_projector),
            "expert_trainable":      count(self.action_expert),
            "total_trainable":       count(self),
            "dual_encoder_frozen":   count_all(self.dual_encoder),
            "total_params":          count_all(self),
        }

    def print_parameter_summary(self):
        stats = self.count_parameters()
        print("\n" + "="*55)
        print("  GST-VLA Parameter Summary")
        print("="*55)
        print(f"  GST (novel):              {stats['gst_trainable']/1e6:.1f}M  [TRAINABLE]")
        print(f"  VLM Projector (W_proj):   {stats['projector_trainable']/1e6:.1f}M  [TRAINABLE]")
        print(f"  Action Expert:            {stats['expert_trainable']/1e6:.1f}M  [TRAINABLE]")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Total Trainable:          {stats['total_trainable']/1e6:.1f}M")
        print(f"  Frozen Encoders:          {stats['dual_encoder_frozen']/1e6:.1f}M")
        print(f"  (VLM: ~7B, not shown if use_mock=True)")
        print("="*55 + "\n")


# ─────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building GST-VLA (mock mode)...")
    model = GSTVLA(
        use_mock_encoders=True,
        use_mock_vlm=True,
    )
    model.print_parameter_summary()

    B = 2
    rgb          = torch.randn(B, 3, 224, 224)
    inst_ids     = torch.ones(B, 32, dtype=torch.long)
    attn_mask    = torch.ones(B, 32, dtype=torch.long)
    robot_state  = torch.randn(B, 14)
    gt_actions   = torch.randn(B, 16, 7)

    print("Forward pass...")
    out = model(rgb, inst_ids, attn_mask, robot_state, gt_actions)
    print(f"  Loss: {out['loss'].item():.4f}")
    print(f"  z_spatial shape: {out['z_spatial'].shape}")
    print(f"  h_vlm shape:     {out['h_vlm'].shape}")

    print("\nInference...")
    actions, info = model.predict_action(rgb, inst_ids, attn_mask, robot_state)
    print(f"  Predicted actions shape: {actions.shape}")
    print(f"  mu_3d shape:             {info['mu_3d'].shape}")
    print("\nGST-VLA sanity check PASSED ✓")
