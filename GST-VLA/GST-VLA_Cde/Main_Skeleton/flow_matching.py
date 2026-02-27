"""
Flow Matching Action Expert for GST-VLA
=========================================
TRAINABLE module.

Architecture:
  - 300M params, N=8 transformer layers, d=512
  - Separate expert (MoE-style single expert for ACCV)
  - FiLM conditioning: h_vlm → adaLN, robot_state → FiLM
  - Flow Matching ODE: 10 Euler steps at inference
  - Action Chunking: H=16 steps, λ=0.01 temporal ensemble

Input:
  h_vlm      ∈ R^(B, N, d_vlm)   — VLM hidden states
  s_t        ∈ R^(B, d_state)    — robot state
  noisy_a    ∈ R^(B, H, 7)       — noisy action chunk (during training)
  t          ∈ R^(B,)            — flow time ∈ [0, 1]

Output:
  v_θ(a, t, h_vlm, s_t) ∈ R^(B, H, 7) — predicted velocity field
  a_t        ∈ R^(B, H, 7)       — denoised action at inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ─────────────────────────────────────────────
# FiLM Conditioning (Feature-wise Linear Modulation)
# ─────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    FiLM: y = γ(c) ⊙ x + β(c)
    Conditioned on context vector c.
    """

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gamma_beta = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.gamma_beta.weight)
        nn.init.zeros_(self.gamma_beta.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, *, d_model)
        c: (B, d_cond)
        """
        # Expand c to match x shape
        while c.dim() < x.dim():
            c = c.unsqueeze(1)  # (B, 1, ..., d_cond)

        gb = self.gamma_beta(c)  # (B, *, 2*d_model)
        gamma, beta = gb.chunk(2, dim=-1)
        x_norm = self.norm(x)
        return (1.0 + gamma) * x_norm + beta


# ─────────────────────────────────────────────
# AdaLN (Adaptive Layer Norm) for VLM conditioning
# ─────────────────────────────────────────────

class AdaLNBlock(nn.Module):
    """
    Adaptive LayerNorm conditioned on pooled VLM features.
    Used in DiT-style transformer blocks.
    
    c_vlm = mean_pool(h_vlm) ∈ R^(B, d_vlm)
    AdaLN: scale, shift = Linear(c_vlm)
    """

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 6 * d_model),  # scale_msa, shift_msa, gate_msa, scale_ffn, shift_ffn, gate_ffn
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    def forward_modulation(self, c: torch.Tensor) -> Tuple:
        """Returns (shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn)"""
        mods = self.adaLN_modulation(c)  # (B, 6*d)
        return mods.chunk(6, dim=-1)


# ─────────────────────────────────────────────
# Sinusoidal Time Embedding
# ─────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal time embedding for flow matching timestep t ∈ [0, 1].
    Maps scalar t → d_model-dim embedding.
    """

    def __init__(self, d_model: int, max_period: int = 10000):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
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
        """
        t: (B,) float in [0, 1]
        Returns: (B, d_model)
        """
        t = t.float()
        args = t.unsqueeze(1) * self.freqs.unsqueeze(0) * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


# ─────────────────────────────────────────────
# Action Expert Transformer Block
# ─────────────────────────────────────────────

class ActionExpertBlock(nn.Module):
    """
    Single transformer block for the Action Expert.
    Uses:
    - Self-attention over action sequence
    - Cross-attention to VLM features
    - FiLM conditioning from robot state
    - AdaLN from VLM context
    """

    def __init__(
        self,
        d_model: int = 512,
        d_vlm: int = 3584,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention to VLM
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
            kdim=d_vlm, vdim=d_vlm,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

        # FiLM for timestep + state conditioning
        self.film = FiLMLayer(d_model, d_model)  # cond = t_emb + state_emb

    def forward(
        self,
        x: torch.Tensor,        # (B, H, d_model) action sequence
        h_vlm: torch.Tensor,    # (B, N, d_vlm)   VLM context
        cond: torch.Tensor,     # (B, d_model)     timestep + state embedding
    ) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        sa, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + sa

        # Cross-attention to VLM
        x_norm = self.norm2(x)
        ca, _ = self.cross_attn(query=x_norm, key=h_vlm, value=h_vlm)
        x = x + ca

        # FiLM conditioning
        x = self.film(x, cond)

        # FFN
        x = x + self.ffn(self.norm3(x))

        return x


# ─────────────────────────────────────────────
# Flow Matching Action Expert (Full)
# ─────────────────────────────────────────────

class FlowMatchingActionExpert(nn.Module):
    """
    Flow Matching Action Expert for GST-VLA.
    
    Learns velocity field: v_θ(a_t, t | h_vlm, s_t)
    at training time via:
        L_flow = E[‖v_θ - (a_1 - a_0)‖²]
    
    where:
        a_0 ~ N(0, I) noise
        a_1 = ground truth action
        a_t = t*a_1 + (1-t)*a_0  (linear interpolation)
    
    At inference: integrate ODE with Euler steps.

    Args:
        d_model:    expert hidden dim (512)
        d_vlm:      VLM feature dim (3584 for Qwen2.5-VL-7B)
        d_state:    robot state dim (default 7: q,dq,p,ee,gripper)
        H:          action chunk horizon (16)
        d_action:   per-step action dim (7: Δpose(6)+gripper(1))
        n_layers:   number of transformer blocks (8)
        n_heads:    attention heads (8)
        n_euler:    Euler integration steps at inference (10)
        lambda_ens: temporal ensemble weight (0.01)
    """

    def __init__(
        self,
        d_model: int = 512,
        d_vlm: int = 3584,
        d_state: int = 14,   # joint pos + vel
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

        # Action input embedding: (B, H, d_action) → (B, H, d_model)
        self.action_embed = nn.Linear(d_action, d_model)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(d_model)

        # Robot state embedding
        self.state_embed = nn.Sequential(
            nn.Linear(d_state, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # VLM conditioning: pool + project to d_model
        self.vlm_pool = nn.Sequential(
            nn.Linear(d_vlm, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ActionExpertBlock(d_model=d_model, d_vlm=d_vlm, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output head: velocity prediction
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_action)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Learnable action chunk positional encoding
        self.action_pos_emb = nn.Parameter(torch.randn(1, H, d_model) * 0.02)

    def forward(
        self,
        noisy_a: torch.Tensor,   # (B, H, d_action) noisy action
        t: torch.Tensor,          # (B,) timestep in [0, 1]
        h_vlm: torch.Tensor,      # (B, N, d_vlm) VLM features
        s_t: torch.Tensor,        # (B, d_state) robot state
    ) -> torch.Tensor:
        """
        Predict velocity field v_θ(a_t, t, h_vlm, s_t).
        
        Returns:
            v: (B, H, d_action) predicted velocity
        """
        B = noisy_a.shape[0]

        # Embed action sequence
        x = self.action_embed(noisy_a) + self.action_pos_emb  # (B, H, d_model)

        # Conditioning: timestep + state
        t_emb = self.time_embed(t)         # (B, d_model)
        s_emb = self.state_embed(s_t)      # (B, d_model)
        cond  = t_emb + s_emb              # (B, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, h_vlm, cond)

        # Output velocity
        v = self.out_proj(self.out_norm(x))  # (B, H, d_action)
        return v

    # ─── Training ─────────────────────────────

    def compute_flow_loss(
        self,
        a1: torch.Tensor,    # (B, H, d_action) ground truth action
        h_vlm: torch.Tensor,
        s_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conditional Flow Matching loss:
            L = E_{t, a_0} [ ‖v_θ(a_t, t) - (a_1 - a_0)‖² ]
        
        where a_t = t * a_1 + (1-t) * a_0, t ~ Uniform[0,1]
        """
        B, H, D = a1.shape
        device = a1.device

        # Sample noise a_0 ~ N(0, I)
        a0 = torch.randn_like(a1)

        # Sample timestep t ~ U[0.01, 0.99]
        t = torch.rand(B, device=device) * 0.98 + 0.01

        # Interpolate: a_t = t * a_1 + (1-t) * a_0
        t_exp = t.view(B, 1, 1)
        a_t = t_exp * a1 + (1 - t_exp) * a0

        # Target velocity (conditional flow)
        v_target = a1 - a0  # (B, H, D)

        # Predict
        v_pred = self.forward(a_t, t, h_vlm, s_t)  # (B, H, D)

        loss = F.mse_loss(v_pred, v_target)
        return loss

    # ─── Inference ────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        h_vlm: torch.Tensor,
        s_t: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate action chunk by integrating ODE with Euler method.
        
        a_0 ~ N(0, I)
        a_{t+Δt} = a_t + Δt * v_θ(a_t, t)  for t: 0 → 1
        
        Returns:
            a_1: (B, H, d_action) predicted action chunk
        """
        n_steps = n_steps or self.n_euler
        B = h_vlm.shape[0]
        device = h_vlm.device

        # Start from noise
        a = torch.randn(B, self.H, self.d_action, device=device)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.forward(a, t, h_vlm, s_t)
            a = a + dt * v

        return a  # (B, H, d_action)

    @torch.no_grad()
    def sample_with_temporal_ensemble(
        self,
        h_vlm_list: list,
        s_t_list: list,
    ) -> torch.Tensor:
        """
        Temporal ensemble: average overlapping predictions weighted by λ.
        
        For H=16 horizon with λ=0.01:
            w_i = exp(-λ * i)  for i=0,...,H-1
        
        Returns:
            a_next: (B, d_action) next action to execute
        """
        weights = torch.exp(-self.lambda_ens * torch.arange(self.H))
        weights = weights / weights.sum()

        assert len(h_vlm_list) == len(s_t_list)
        n = len(h_vlm_list)
        device = h_vlm_list[-1].device

        action_sum = None
        weight_sum = 0.0

        for i, (h, s) in enumerate(zip(h_vlm_list, s_t_list)):
            chunk = self.sample(h, s)  # (B, H, d_action)
            # Take the i-th step from this chunk
            step_idx = n - 1 - i
            if step_idx < self.H:
                w = weights[step_idx].to(device)
                a_step = chunk[:, step_idx, :]  # (B, d_action)
                if action_sum is None:
                    action_sum = w * a_step
                else:
                    action_sum = action_sum + w * a_step
                weight_sum += w

        return action_sum / weight_sum


# Allow optional n_steps
from typing import Optional
