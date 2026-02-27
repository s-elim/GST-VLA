import math
import torch
import torch.nn as nn
from typing import Optional

class SinusoidalPositionEmbedding(nn.Module):
    """Standard sinusoidal time embedding for the Flow Matching ODE time step."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ModulatedDiTBlock(nn.Module):
    """
    A Diffusion Transformer (DiT) block using Adaptive Layer Norm (AdaLN).
    This injects the combined (Time + VLM + State) conditioning into every layer.
    """
    def __init__(self, hidden_size: int, num_heads: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Standard MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # AdaLN modulation parameters: scale and shift for both norms + scale for residuals
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Generate modulation parameters from the conditioning vector
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        
        # Modulated Self-Attention
        norm_x = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Modulated MLP
        norm_x2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(norm_x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class FlowMatchingExpert(nn.Module):
    def __init__(
        self, 
        action_dim: int = 7,       # e.g., Delta XYZ (3) + Euler angles (3) + Gripper (1)
        chunk_size: int = 16,      # H=16 from your diagram
        embed_dim: int = 512,      # d=512 from your diagram
        vlm_dim: int = 3584,       # Qwen2.5-VL hidden dimension
        state_dim: int = 14,       # q, dq, x_ee, gripper
        num_layers: int = 12,      # Scale N to hit ~300M parameters
        num_heads: int = 16
    ):
        """
        Flow Matching Action Expert using a DiT architecture.
        Predicts the vector field v(x_t, t, c_vlm, s_t).
        """
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim
        
        # 1. Input Projections
        # Map noisy actions x_t to transformer hidden dimension
        self.action_proj = nn.Linear(action_dim, embed_dim)
        # Sequence positional embedding for the action chunk
        self.pos_embed = nn.Parameter(torch.randn(1, chunk_size, embed_dim) * 0.02)
        
        # 2. Conditioning Encoders
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.vlm_pooler = nn.Sequential(
            nn.Linear(vlm_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Total conditioning dimension combining time, vlm context, and robot state
        cond_dim = embed_dim * 3
        
        # 3. Transformer Blocks (The 300M parameter core)
        self.blocks = nn.ModuleList([
            ModulatedDiTBlock(embed_dim, num_heads, cond_dim) 
            for _ in range(num_layers)
        ])
        
        # 4. Output Head
        self.final_layer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, action_dim)
        )
        
        # Initialize final layer weights to zero for flow matching stability
        nn.init.zeros_(self.final_layer[-1].weight)
        nn.init.zeros_(self.final_layer[-1].bias)

    def forward(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        vlm_hidden_states: torch.Tensor,
        robot_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, 16, 7) noisy action trajectory at time t
            t: (B,) time steps t ~ U[0, 1]
            vlm_hidden_states: (B, Seq_Len, 3584) outputs from frozen Qwen
            robot_state: (B, 14) current proprioception
        Returns:
            v_pred: (B, 16, 7) predicted vector field
        """
        B = x_t.shape[0]
        
        # 1. Prepare sequence input: (B, chunk_size, embed_dim)
        x = self.action_proj(x_t) + self.pos_embed
        
        # 2. Prepare Conditioning
        t_emb = self.time_mlp(t) # (B, embed_dim)
        
        # Pool VLM hidden states: average pooling over the spatial/text tokens
        # c_vlm <- pool(N_v, h_vlm) as per the diagram
        c_vlm = self.vlm_pooler(vlm_hidden_states.mean(dim=1)) # (B, embed_dim)
        
        # Encode robot state
        s_t = self.state_proj(robot_state) # (B, embed_dim)
        
        # Concatenate conditioning vectors: (B, embed_dim * 3)
        cond = torch.cat([t_emb, c_vlm, s_t], dim=-1)
        
        # 3. Apply DiT Blocks
        for block in self.blocks:
            x = block(x, cond)
            
        # 4. Predict Vector Field
        v_pred = self.final_layer(x) # (B, chunk_size, action_dim)
        
        return v_pred