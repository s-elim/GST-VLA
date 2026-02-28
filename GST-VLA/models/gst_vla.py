"""
DEAD-VLA: Full Model Assembly
==============================
Depth-Enhanced Affordance-Driven Vision-Language-Action Model
ACCV 2026 — via Chain-of-Thought Spatial Reasoning and Gaussian Spatial Tokenization

Pipeline (5 stages):
    ENCODE:   RGB + Depth ──[FROZEN]──► f_sem (B,256,1152) + D̂ (B,H,W)
    TOKENIZE: GST ─────────[TRAINABLE]► z_spatial (B,128,512)
    REASON:   VLM ─────────[FROZEN]───► h_vlm (B,N,3584)
              DA-CoT ───────[TRAINABLE]► h_cot (B,N_q+N,3584)
    POLICY:   FlowExpert ──[TRAINABLE]► v_θ → a_t (B,16,7)
    EXECUTE:  a_t ∈ R^(16×7)  [Δpose(6) + gripper(1)] × H

VLM input sequence: [z_spatial | l_tokens | MLP(s_t)]

Training objectives:
    L = L_flow + λ_CoT * L_CoT + λ_depth * L_depth

3-Stage training:
    S1: GST + Expert + DA-CoT (frozen VLM, SigLIP, DepthV2)
    S2: +LoRA on VLM
    S3: Full fine-tune

Trainable parameters (~60M for S1, ~330M total):
    GST:           ~15M
    W_proj:         ~2M
    State encoder:  ~3M
    DA-CoT:        ~10M
    Flow Expert:  ~300M
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from models.encoders import DualEncoder
from models.gst import GaussianSpatialTokenizer, GSTtoVLMProjector
from models.vlm_wrapper import QwenVLMWrapper
from models.flow_matching import FlowMatchingActionExpert
from models.da_cot import DepthAwareCoTReasoner


class GSTVLA(nn.Module):
    """
    DEAD-VLA full pipeline.

    Trainable: GST, VLM projector, state encoder, DA-CoT, action expert.
    Frozen:    SigLIP, Depth Anything V2, Qwen2.5-VL-7B.
    """

    def __init__(
        self,
        # Encoder / GST config
        d_sem: int = 1152,
        d_gst: int = 512,
        N_g: int = 128,
        fourier_bands: int = 16,
        img_size: int = 224,
        patch_size: int = 14,
        # VLM config
        d_vlm: int = 3584,
        vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        # Robot state + action
        d_state: int = 14,
        H: int = 16,
        d_action: int = 7,
        # Action expert config
        n_expert_layers: int = 8,
        n_euler_steps: int = 10,
        # DA-CoT config
        n_cot_queries: int = 16,
        d_cot: int = 512,
        K_obj: int = 8,
        K_grasp: int = 4,
        K_rel: int = 8,
        K_wp: int = 8,
        # Runtime flags
        use_mock_encoders: bool = False,
        use_mock_vlm: bool = False,
        depth_pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.H        = H
        self.d_action = d_action
        self.d_vlm    = d_vlm

        # ── ENCODE: Frozen Dual Encoder ──────────────────────
        self.dual_encoder = DualEncoder(
            use_mock=use_mock_encoders,
            depth_pretrained_path=depth_pretrained_path,
        )
        for p in self.dual_encoder.parameters():
            p.requires_grad_(False)

        # ── TOKENIZE: GST (novel, trainable) ─────────────────
        self.gst = GaussianSpatialTokenizer(
            d_sem=d_sem, d_gst=d_gst, N_g=N_g,
            fourier_bands=fourier_bands,
            img_size=img_size, patch_size=patch_size,
        )

        # ── TOKENIZE: Cross-attention projector W_proj ────────
        self.vlm_projector = GSTtoVLMProjector(d_gst=d_gst, d_vlm=d_vlm)

        # ── TOKENIZE: State encoder  MLP(s_t) → 1 VLM token ──
        self.state_encoder = nn.Sequential(
            nn.Linear(d_state, d_vlm // 4),
            nn.SiLU(),
            nn.Linear(d_vlm // 4, d_vlm),
            nn.LayerNorm(d_vlm),
        )

        # ── REASON: Frozen VLM ────────────────────────────────
        self.vlm = QwenVLMWrapper(model_name=vlm_model_name, use_mock=use_mock_vlm)

        # ── REASON: DA-CoT (trainable, supervised) ───────────
        self.da_cot = DepthAwareCoTReasoner(
            d_vlm=d_vlm, d_gst=d_gst, d_cot=d_cot,
            n_queries=n_cot_queries,
            K_obj=K_obj, K_grasp=K_grasp, K_rel=K_rel, K_wp=K_wp,
        )

        # ── POLICY: Flow Matching Action Expert (trainable) ───
        self.action_expert = FlowMatchingActionExpert(
            d_model=512, d_vlm=d_vlm,
            d_state=d_state, H=H, d_action=d_action,
            n_layers=n_expert_layers, n_euler=n_euler_steps,
        )

    # ─────────────────────────────────────────────────────
    # Forward pass (training)
    # ─────────────────────────────────────────────────────

    def forward(
        self,
        rgb: torch.Tensor,              # (B, 3, 224, 224)
        instruction_ids: torch.Tensor,  # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
        robot_state: torch.Tensor,      # (B, d_state)
        gt_actions: torch.Tensor,       # (B, H, d_action)
        camera_K: Optional[torch.Tensor] = None,   # (B, 3, 3)
        cot_targets: Optional[Dict] = None,         # optional GT reasoning
    ) -> Dict[str, torch.Tensor]:
        """
        Full training forward pass.

        Returns dict with:
            loss        — total training loss
            loss_flow   — flow matching loss
            loss_cot    — DA-CoT reasoning loss (0 if no targets)
            z_spatial   — (B, N_g, d_gst)
            h_vlm       — (B, N_g+L+1, d_vlm)
            h_cot       — (B, N_q+N_g+L+1, d_vlm)
            reasoning   — dict of 4 spatial reasoning predictions
            gst_aux     — GST intermediate outputs
        """
        # ── 1. Encode (frozen) ────────────────────────────────
        with torch.no_grad():
            f_sem, depth_map = self.dual_encoder(rgb)

        # ── 2. GST → spatial tokens ──────────────────────────
        z_spatial, gst_aux = self.gst(
            f_sem=f_sem, depth_map=depth_map, K=camera_K
        )  # (B, N_g, d_gst)

        # ── 3. Project to VLM space ───────────────────────────
        spatial_tokens_vlm = self.vlm_projector(z_spatial)   # (B, N_g, d_vlm)

        # ── 4. State encoding → 1 extra token ────────────────
        state_token = self.state_encoder(robot_state).unsqueeze(1)  # (B, 1, d_vlm)

        # ── 5. VLM reasoning: [spatial | language | state] ───
        # NOTE: no torch.no_grad() here — frozen VLM params have requires_grad=False,
        # so they won't be updated, but gradients still flow THROUGH the VLM to
        # reach state_encoder and vlm_projector (needed for S1 training).
        h_vlm = self.vlm(
            spatial_tokens=spatial_tokens_vlm,
            input_ids=instruction_ids,
            attention_mask=attention_mask,
            state_token=state_token,
        )  # (B, N_g + L + 1, d_vlm)

        # ── 6. DA-CoT: enrich features + structured reasoning ─
        h_cot, reasoning = self.da_cot(h_vlm, z_spatial)
        # h_cot: (B, n_cot_queries + N_g + L + 1, d_vlm)

        # ── 7. DA-CoT loss ────────────────────────────────────
        loss_cot = self.da_cot.compute_cot_loss(reasoning, cot_targets)

        # ── 8. Flow matching loss ─────────────────────────────
        loss_flow = self.action_expert.compute_flow_loss(
            a1=gt_actions, h_vlm=h_cot, s_t=robot_state,
        )

        # Combined loss (full weighting handled in GSTVLALoss)
        loss = loss_flow + 0.1 * loss_cot

        return {
            "loss":      loss,
            "loss_flow": loss_flow,
            "loss_cot":  loss_cot,
            "z_spatial": z_spatial,
            "h_vlm":     h_vlm,
            "h_cot":     h_cot,
            "reasoning": reasoning,
            "gst_aux":   gst_aux,
        }

    # ─────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_action(
        self,
        rgb: torch.Tensor,
        instruction_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        robot_state: torch.Tensor,
        camera_K: Optional[torch.Tensor] = None,
        n_euler_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Inference: predict action chunk a_t ∈ R^(B, H, 7).

        Returns:
            actions: (B, H, 7)
            info:    dict with 3D spatial info and reasoning outputs
        """
        f_sem, depth_map       = self.dual_encoder(rgb)
        z_spatial, gst_aux     = self.gst(f_sem=f_sem, depth_map=depth_map, K=camera_K)
        spatial_tokens_vlm     = self.vlm_projector(z_spatial)
        state_token            = self.state_encoder(robot_state).unsqueeze(1)

        h_vlm = self.vlm(
            spatial_tokens=spatial_tokens_vlm,
            input_ids=instruction_ids,
            attention_mask=attention_mask,
            state_token=state_token,
        )

        h_cot, reasoning = self.da_cot(h_vlm, z_spatial)

        actions = self.action_expert.sample(
            h_vlm=h_cot, s_t=robot_state, n_steps=n_euler_steps,
        )  # (B, H, 7)

        info = {
            "z_spatial":  z_spatial,
            "mu_3d":      gst_aux["mu_3d"],
            "alpha":      gst_aux["alpha"],
            "depth_map":  depth_map,
            "h_cot":      h_cot,
            "reasoning":  reasoning,
        }
        return actions, info

    # ─────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        def n_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        def n_all(m):       return sum(p.numel() for p in m.parameters())
        return {
            "gst":            n_trainable(self.gst),
            "vlm_projector":  n_trainable(self.vlm_projector),
            "state_encoder":  n_trainable(self.state_encoder),
            "da_cot":         n_trainable(self.da_cot),
            "action_expert":  n_trainable(self.action_expert),
            "total_trainable": n_trainable(self),
            "dual_encoder_frozen": n_all(self.dual_encoder),
        }

    def print_parameter_summary(self):
        s = self.count_parameters()
        print("\n" + "=" * 60)
        print("  DEAD-VLA Parameter Summary  (ACCV 2026)")
        print("=" * 60)
        print(f"  [C1] GST (novel):              {s['gst']/1e6:6.2f}M  [TRAINABLE]")
        print(f"       W_proj (GST→VLM):          {s['vlm_projector']/1e6:6.2f}M  [TRAINABLE]")
        print(f"       State encoder MLP(s_t):    {s['state_encoder']/1e6:6.2f}M  [TRAINABLE]")
        print(f"  [C2] DA-CoT Reasoner:           {s['da_cot']/1e6:6.2f}M  [TRAINABLE]")
        print(f"  [C3] Flow Matching Expert:      {s['action_expert']/1e6:6.2f}M  [TRAINABLE]")
        print(f"  {'─'*52}")
        print(f"       Total Trainable:            {s['total_trainable']/1e6:6.2f}M")
        print(f"       Frozen Encoders:            {s['dual_encoder_frozen']/1e6:6.2f}M")
        print(f"       VLM (Qwen2.5-VL-7B):       ~7000M  [FROZEN]")
        print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building DEAD-VLA (mock mode)...")
    model = GSTVLA(use_mock_encoders=True, use_mock_vlm=True)
    model.print_parameter_summary()

    B = 2
    rgb         = torch.randn(B, 3, 224, 224)
    inst_ids    = torch.ones(B, 32, dtype=torch.long)
    attn_mask   = torch.ones(B, 32, dtype=torch.long)
    robot_state = torch.randn(B, 14)
    gt_actions  = torch.randn(B, 16, 7)

    out = model(rgb, inst_ids, attn_mask, robot_state, gt_actions)
    print(f"  loss_flow : {out['loss_flow'].item():.4f}")
    print(f"  loss_cot  : {out['loss_cot'].item():.4f}")
    print(f"  z_spatial : {out['z_spatial'].shape}")
    print(f"  h_cot     : {out['h_cot'].shape}")

    actions, info = model.predict_action(rgb, inst_ids, attn_mask, robot_state)
    print(f"  actions   : {actions.shape}")
    print("\nDEAD-VLA sanity check PASSED ✓")
