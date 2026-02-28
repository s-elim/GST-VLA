"""
Flow Matching Action Expert for DEAD-VLA
==========================================
TRAINABLE policy module — ACCV 2026

Architecture:
  - 300M params, N=8 transformer layers, d=512
  - Separate expert (MoE-style single expert)
  - Conditioning:  h_vlm → adaLN,  robot_state → FiLM
  - Flow Matching ODE: v_θ(a_t, t, h_vlm, s_t)
  - 10 Euler steps at inference
  - Action Chunking: H=16,  temporal ensemble λ=0.01

Input:
    h_vlm   ∈ R^(B, N, d_vlm)    VLM/CoT hidden states
    s_t     ∈ R^(B, d_state)     robot state
    a_noisy ∈ R^(B, H, 7)        noisy action chunk (training)
    t       ∈ R^(B,)              flow time ∈ [0, 1]

Output:
    v_θ(a, t, h, s) ∈ R^(B, H, 7)  predicted velocity field
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


# ─────────────────────────────────────────────
# FiLM Conditioning
# ─────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: y = γ(c) ⊙ norm(x) + β(c)"""

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gamma_beta = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.gamma_beta.weight)
        nn.init.zeros_(self.gamma_beta.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, *, d_model)  c: (B, d_cond)"""
        while c.dim() < x.dim():
            c = c.unsqueeze(1)
        gb = self.gamma_beta(c)
        gamma, beta = gb.chunk(2, dim=-1)
        return (1.0 + gamma) * self.norm(x) + beta


# ─────────────────────────────────────────────
# Sinusoidal Timestep Embedding
# ─────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """Maps scalar t ∈ [0,1] → d_model-dim sinusoidal embedding."""

    def __init__(self, d_model: int, max_period: int = 10000):
        super().__init__()
        assert d_model % 2 == 0
        half = d_model // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, d_model)"""
        args = t.float().unsqueeze(1) * self.freqs.unsqueeze(0) * 1000.0
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


# ─────────────────────────────────────────────
# Action Expert Transformer Block
# ─────────────────────────────────────────────

class ActionExpertBlock(nn.Module):
    """
    Single transformer block for the action expert:
        - Self-attention over action chunk
        - Cross-attention to VLM/CoT features
        - FiLM conditioning from (timestep + state)
        - FFN
    """

    def __init__(
        self,
        d_model: int = 512,
        d_vlm: int = 3584,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1      = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
            kdim=d_vlm, vdim=d_vlm,
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.film = FiLMLayer(d_model, d_model)  # cond = t_emb + s_emb

    def forward(
        self,
        x: torch.Tensor,        # (B, H, d_model)
        h_vlm: torch.Tensor,    # (B, N, d_vlm)
        cond: torch.Tensor,     # (B, d_model)
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(*[self.norm1(x)] * 3)[0]
        # Cross-attention to VLM
        x = x + self.cross_attn(self.norm2(x), h_vlm, h_vlm)[0]
        # FiLM conditioning
        x = self.film(x, cond)
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


# ─────────────────────────────────────────────
# Flow Matching Action Expert
# ─────────────────────────────────────────────

class FlowMatchingActionExpert(nn.Module):
    """
    Conditional flow matching policy:
        v_θ(a_t, t | h_vlm, s_t) ∈ R^(B, H, d_action)

    Training loss (CFM):
        L_flow = E_{t,a_0} [ ‖v_θ(a_t, t) − (a_1 − a_0)‖² ]

    Inference — Euler ODE integration:
        a_{i+1} = a_i + Δt · v_θ(a_i, t_i)   for t: 0 → 1
    """

    def __init__(
        self,
        d_model: int = 512,
        d_vlm: int = 3584,
        d_state: int = 14,
        H: int = 16,
        d_action: int = 7,
        n_layers: int = 8,
        n_heads: int = 8,
        n_euler: int = 10,
        lambda_ens: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.H          = H
        self.d_action   = d_action
        self.n_euler    = n_euler
        self.lambda_ens = lambda_ens
        self.d_model    = d_model

        self.action_embed = nn.Linear(d_action, d_model)
        self.time_embed   = TimestepEmbedding(d_model)
        self.state_embed  = nn.Sequential(
            nn.Linear(d_state, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            ActionExpertBlock(d_model=d_model, d_vlm=d_vlm, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_action)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.action_pos_emb = nn.Parameter(torch.randn(1, H, d_model) * 0.02)

    def forward(
        self,
        noisy_a: torch.Tensor,   # (B, H, d_action)
        t: torch.Tensor,          # (B,)
        h_vlm: torch.Tensor,      # (B, N, d_vlm)
        s_t: torch.Tensor,        # (B, d_state)
    ) -> torch.Tensor:
        """Predict velocity v_θ(a_t, t, h_vlm, s_t). Returns (B, H, d_action)."""
        x    = self.action_embed(noisy_a) + self.action_pos_emb
        cond = self.time_embed(t) + self.state_embed(s_t)

        for block in self.blocks:
            x = block(x, h_vlm, cond)

        return self.out_proj(self.out_norm(x))  # (B, H, d_action)

    def compute_flow_loss(
        self,
        a1: torch.Tensor,    # (B, H, d_action) ground truth action
        h_vlm: torch.Tensor,
        s_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conditional Flow Matching loss:
            L = E_{t,a_0} [ ‖v_θ(a_t, t) − (a_1 − a_0)‖² ]
        """
        B = a1.shape[0]
        device = a1.device

        a0 = torch.randn_like(a1)
        t  = torch.rand(B, device=device) * 0.98 + 0.01
        t_exp = t.view(B, 1, 1)

        a_t      = t_exp * a1 + (1 - t_exp) * a0  # interpolate
        v_target = a1 - a0                          # target velocity

        v_pred = self.forward(a_t, t, h_vlm, s_t)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(
        self,
        h_vlm: torch.Tensor,
        s_t: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate action chunk via Euler ODE integration:
            a_{i+1} = a_i + Δt · v_θ(a_i, t_i)
        Returns: (B, H, d_action)
        """
        n_steps = n_steps or self.n_euler
        B = h_vlm.shape[0]
        device = h_vlm.device

        a  = torch.randn(B, self.H, self.d_action, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            a = a + dt * self.forward(a, t, h_vlm, s_t)

        return a  # (B, H, d_action)

    @torch.no_grad()
    def sample_with_temporal_ensemble(
        self,
        h_vlm_list: List[torch.Tensor],
        s_t_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Temporal ensemble: exponentially-weighted average of overlapping predictions.
            w_i = exp(−λ * i),  λ=0.01
        Returns: (B, d_action) next action to execute.
        """
        weights = torch.exp(-self.lambda_ens * torch.arange(self.H, dtype=torch.float32))
        weights = weights / weights.sum()

        n = len(h_vlm_list)
        device = h_vlm_list[-1].device

        action_sum = None
        weight_sum = 0.0

        for i, (h, s) in enumerate(zip(h_vlm_list, s_t_list)):
            chunk    = self.sample(h, s)        # (B, H, d_action)
            step_idx = n - 1 - i
            if step_idx < self.H:
                w = weights[step_idx].to(device)
                a_step = chunk[:, step_idx, :]  # (B, d_action)
                action_sum = w * a_step if action_sum is None else action_sum + w * a_step
                weight_sum += w

        return action_sum / weight_sum
