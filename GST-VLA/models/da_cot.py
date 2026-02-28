"""
DA-CoT: Depth-Aware Chain-of-Thought Reasoning Module
=======================================================
ACCV 2026 — Novel Contribution C2

Processes VLM hidden states + 3D Gaussian tokens to produce structured
spatial reasoning outputs, supervised via:
    L_CoT = Σ SmoothL1(pred_reasoning, gt_reasoning)

Four structured reasoning outputs (supervised when GT available):
    ① Object grounding:   3D center coordinates  ∈ R^(B, K_obj, 3)
    ② Grasp affordance:   contact point locations ∈ R^(B, K_grasp, 3)
    ③ Spatial relations:  metric 3D relationships ∈ R^(B, K_rel, 6)
    ④ Motion plan:        SE(3) waypoints         ∈ R^(B, K_wp, 7)

Architecture:
    - N_q learnable CoT queries cross-attend to [h_vlm | z_spatial]
    - 4 MLP prediction heads decode from pooled query features
    - Output h_cot = [CoT_tokens | h_vlm] enriches action expert input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class CoTCrossAttentionBlock(nn.Module):
    """
    Single decoder block for CoT queries attending to context features.
    Pre-norm style with self-attention + cross-attention + FFN.
    """

    def __init__(self, d_model: int, d_ctx: int, n_heads: int = 8):
        super().__init__()
        # Self-attention over queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention: queries attend to context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True,
            kdim=d_ctx, vdim=d_ctx,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,    # (B, Q, d_model)
        context: torch.Tensor,    # (B, N, d_ctx)
    ) -> torch.Tensor:
        # Self-attention
        q = self.norm1(queries)
        sa, _ = self.self_attn(q, q, q)
        queries = queries + sa

        # Cross-attention to context
        q = self.norm2(queries)
        ca, _ = self.cross_attn(q, context, context)
        queries = queries + ca

        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class DepthAwareCoTReasoner(nn.Module):
    """
    DA-CoT: Depth-Aware Chain-of-Thought Reasoner.

    Takes:
        h_vlm:     (B, N, d_vlm)     VLM hidden states
        z_spatial: (B, N_g, d_gst)   3D Gaussian tokens (depth-aware context)

    Outputs:
        h_cot:    (B, N_q + N, d_vlm) enriched features for action expert
        reasoning: dict with 4 spatial reasoning predictions

    Usage:
        h_cot, reasoning = da_cot(h_vlm, z_spatial)
        loss_cot = da_cot.compute_cot_loss(reasoning, cot_targets)
    """

    def __init__(
        self,
        d_vlm: int = 3584,
        d_gst: int = 512,
        d_cot: int = 512,
        n_heads: int = 8,
        n_queries: int = 16,       # number of learnable CoT query tokens
        n_cot_layers: int = 2,     # decoder depth
        # Reasoning head output sizes
        K_obj: int = 8,            # max objects to ground (3D centers)
        K_grasp: int = 4,          # contact points for grasping
        K_rel: int = 8,            # spatial relation pairs
        K_wp: int = 8,             # SE(3) waypoints
    ):
        super().__init__()
        self.n_queries = n_queries
        self.d_vlm = d_vlm
        self.K_obj   = K_obj
        self.K_grasp = K_grasp
        self.K_rel   = K_rel
        self.K_wp    = K_wp

        # Learnable CoT query tokens
        self.cot_queries = nn.Parameter(torch.randn(n_queries, d_cot) * 0.02)

        # Project VLM features to d_cot for cross-attention
        self.vlm_proj = nn.Linear(d_vlm, d_cot)

        # Project 3D spatial tokens to d_cot (depth awareness)
        self.spatial_proj = nn.Linear(d_gst, d_cot)

        # CoT decoder layers
        self.cot_layers = nn.ModuleList([
            CoTCrossAttentionBlock(d_cot, d_cot, n_heads)
            for _ in range(n_cot_layers)
        ])

        # Project CoT queries back to d_vlm for prepending to h_vlm
        self.out_proj = nn.Linear(d_cot, d_vlm)
        self.out_norm = nn.LayerNorm(d_vlm)

        # ── 4 Structured Reasoning Heads ─────────────────────
        # ① Object grounding: predict 3D center for each object slot
        self.obj_head = nn.Sequential(
            nn.Linear(d_cot, d_cot),
            nn.GELU(),
            nn.Linear(d_cot, K_obj * 3),
        )
        # ② Grasp affordance: predict contact point locations
        self.grasp_head = nn.Sequential(
            nn.Linear(d_cot, d_cot),
            nn.GELU(),
            nn.Linear(d_cot, K_grasp * 3),
        )
        # ③ Spatial relations: 6-DoF relative pose between object pairs
        self.rel_head = nn.Sequential(
            nn.Linear(d_cot, d_cot),
            nn.GELU(),
            nn.Linear(d_cot, K_rel * 6),
        )
        # ④ Motion plan: SE(3) waypoints = pos(3) + axis-angle(3) + gripper(1)
        self.plan_head = nn.Sequential(
            nn.Linear(d_cot, d_cot),
            nn.GELU(),
            nn.Linear(d_cot, K_wp * 7),
        )

    def forward(
        self,
        h_vlm: torch.Tensor,        # (B, N, d_vlm)
        z_spatial: torch.Tensor,    # (B, N_g, d_gst)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            h_cot:    (B, N_q + N, d_vlm)  enriched context for action expert
            reasoning: dict of 4 spatial reasoning predictions
        """
        B = h_vlm.shape[0]

        # Project to d_cot space
        vlm_ctx     = self.vlm_proj(h_vlm)          # (B, N, d_cot)
        spatial_ctx = self.spatial_proj(z_spatial)   # (B, N_g, d_cot)

        # Combine VLM + 3D spatial context for cross-attention
        context = torch.cat([vlm_ctx, spatial_ctx], dim=1)  # (B, N+N_g, d_cot)

        # Expand learnable queries over batch
        queries = self.cot_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N_q, d_cot)

        # CoT decoder: queries attend to [VLM + 3D] context
        for layer in self.cot_layers:
            queries = layer(queries, context)  # (B, N_q, d_cot)

        # Project queries back to VLM space
        q_vlm = self.out_norm(self.out_proj(queries))  # (B, N_q, d_vlm)

        # Enrich h_vlm by prepending CoT tokens
        h_cot = torch.cat([q_vlm, h_vlm], dim=1)  # (B, N_q + N, d_vlm)

        # ── Prediction heads ─────────────────────────────────
        # Average-pool queries as shared representation
        q_pool = queries.mean(dim=1)  # (B, d_cot)

        reasoning = {
            "obj_grounding":    self.obj_head(q_pool).view(B, self.K_obj, 3),
            "grasp_affordance": self.grasp_head(q_pool).view(B, self.K_grasp, 3),
            "spatial_relations": self.rel_head(q_pool).view(B, self.K_rel, 6),
            "motion_plan":      self.plan_head(q_pool).view(B, self.K_wp, 7),
        }

        return h_cot, reasoning

    def compute_cot_loss(
        self,
        reasoning: Dict[str, torch.Tensor],
        cot_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute L_CoT = mean SmoothL1(pred, gt) over available annotations.

        cot_targets keys (all optional):
            'obj_grounding':     (B, K_obj, 3)
            'grasp_affordance':  (B, K_grasp, 3)
            'spatial_relations': (B, K_rel, 6)
            'motion_plan':       (B, K_wp, 7)

        Returns zero when no targets are provided (unsupervised setting).
        """
        device = next(iter(reasoning.values())).device

        if cot_targets is None:
            return torch.zeros(1, device=device).squeeze()

        losses = []
        for key in ["obj_grounding", "grasp_affordance", "spatial_relations", "motion_plan"]:
            if key in cot_targets and cot_targets[key] is not None:
                pred = reasoning[key]
                gt   = cot_targets[key].to(device)
                # Align along slot dimension in case sizes differ
                min_k = min(pred.shape[1], gt.shape[1])
                losses.append(F.smooth_l1_loss(pred[:, :min_k], gt[:, :min_k]))

        if not losses:
            return torch.zeros(1, device=device).squeeze()

        return torch.stack(losses).mean()
