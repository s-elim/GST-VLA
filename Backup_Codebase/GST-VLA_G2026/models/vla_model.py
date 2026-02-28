import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

# Assuming the previous parts are saved in the models/ directory
from models.gst import GaussianSpatialTokenizer
from models.flow_expert import FlowMatchingExpert
from models.qwen_wrapper import FrozenQwenVLWrapper

class GST_VLA(nn.Module):
    def __init__(
        self,
        action_dim: int = 14,
        chunk_size: int = 16,
        vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        GST-VLA: Structured Gaussian Spatial Tokens for 3D-Aware Vision-Language-Action Models
        """
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        # ==========================================
        # 1. FROZEN PERCEPTION BACKBONES (Mocked for skeleton)
        # In production, replace with actual transformers (e.g., SigLIPModel.from_pretrained)
        # ==========================================
        self.semantic_encoder = nn.Linear(3 * 224 * 224, 1152).requires_grad_(False)
        self.depth_expert = nn.Linear(3 * 224 * 224, 1).requires_grad_(False)
        
        # ==========================================
        # 2. TRAINABLE NOVELTY: Gaussian Spatial Tokenizer
        # ==========================================
        self.gst = GaussianSpatialTokenizer(
            semantic_dim=1152,
            embed_dim=3584,
            num_out_tokens=128
        )
        
        # ==========================================
        # 3. FROZEN REASONING CORE: Qwen2.5-VL Wrapper
        # ==========================================
        self.vlm = FrozenQwenVLWrapper(
            model_name=vlm_model_name,
            embed_dim=3584
        )
        
        # ==========================================
        # 4. TRAINABLE POLICY: Flow Matching Expert
        # ==========================================
        self.action_expert = FlowMatchingExpert(
            action_dim=action_dim,
            chunk_size=chunk_size,
            embed_dim=512,
            vlm_dim=3584,
            state_dim=14
        )

    def extract_frozen_features(self, rgb_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the frozen vision models. Wrapped in no_grad to save memory."""
        B = rgb_image.shape[0]
        with torch.no_grad():
            # Mock extraction: RGB -> 256 patches of 1152 dims
            sem_feats = self.semantic_encoder(rgb_image.flatten(1)).view(B, 256, 1152)
            # Mock extraction: RGB -> 16x16 depth map
            depth_map = self.depth_expert(rgb_image.flatten(1)).view(B, 1, 16, 16)
        return sem_feats, depth_map

    def forward(
        self,
        rgb_image: torch.Tensor,
        intrinsics: torch.Tensor,
        text_prompts: List[str],
        robot_state: torch.Tensor,
        actions_target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Main forward pass. 
        If actions_target is provided, returns the Flow Matching MSE loss.
        If actions_target is None, runs Euler integration to predict actions.
        """
        B = rgb_image.shape[0]
        
        # 1. Visual Encoding (Frozen)
        sem_feats, depth_map = self.extract_frozen_features(rgb_image)
        
        # 2. 3D Spatial Tokenization (Trainable)
        # z_spatial shape: (B, 128, 3584)
        z_spatial = self.gst(sem_feats, depth_map, intrinsics)
        
        # 3. VLM Reasoning (Frozen)
        # vlm_hidden shape: (B, Seq_Len, 3584)
        vlm_hidden = self.vlm(text_prompts, z_spatial, self.device)
        
        # 4. Action Policy Execution
        if self.training and actions_target is not None:
            return self.compute_flow_matching_loss(vlm_hidden, robot_state, actions_target)
        else:
            return self.generate_actions(vlm_hidden, robot_state)

    def compute_flow_matching_loss(
        self, 
        vlm_hidden: torch.Tensor, 
        robot_state: torch.Tensor, 
        x_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Optimal Transport Flow Matching loss.
        L = || v_theta(x_t) - (x_1 - x_0) ||^2
        """
        B = x_1.shape[0]
        
        # Sample time t ~ U[0, 1]
        t = torch.rand(B, 1, device=self.device)
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)
        
        # Construct the optimal transport path x_t
        t_expand = t.unsqueeze(-1) # (B, 1, 1) for broadcasting over chunk and action dims
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        # Predict the vector field
        v_pred = self.action_expert(x_t, t, vlm_hidden, robot_state)
        
        # The target vector field in OT Flow Matching is exactly (x_1 - x_0)
        v_target = x_1 - x_0
        
        # Compute MSE Loss
        loss = F.mse_loss(v_pred, v_target)
        
        return loss

    def generate_actions(
        self, 
        vlm_hidden: torch.Tensor, 
        robot_state: torch.Tensor, 
        num_steps: int = 10
    ) -> torch.Tensor:
        """
        Inference via Euler Integration over the learned vector field.
        """
        B = vlm_hidden.shape[0]
        
        # Start from pure Gaussian noise at t=0
        x_t = torch.randn(B, self.chunk_size, self.action_dim, device=self.device)
        
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            # Current time t
            t = torch.full((B, 1), step * dt, device=self.device)
            
            # Query the expert for the vector field (derivative)
            v_pred = self.action_expert(x_t, t, vlm_hidden, robot_state)
            
            # Take an Euler step: x_{t+dt} = x_t + v * dt
            x_t = x_t + v_pred * dt
            
        return x_t # Final predicted action trajectory (x_1)