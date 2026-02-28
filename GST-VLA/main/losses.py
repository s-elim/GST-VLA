"""
Training Losses & 3-Stage Trainer for GST-VLA
===============================================
ACCV 2026

Loss functions:
    L_flow  = E[‖v_θ - (a_1 - a_0)‖²]        (Flow Matching)
    L_depth = SiLog(D̂, D_gt)                   (Depth supervision, optional)
    L_total = L_flow + λ₁*L_depth

3-Stage Training Pipeline:
    S1: Train GST + Depth Expert only
        - Freeze VLM, SigLIP, action expert
        - Loss: L_depth + GST regularization
        
    S2: Add LoRA on VLM + train action expert
        - Freeze SigLIP, Depth Anything
        - Loss: L_flow + L_depth
        
    S3: Full fine-tune
        - Unfreeze GST + projector + expert
        - Optional: unfreeze LoRA
        - Loss: L_total
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────

def flow_matching_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """
    Flow matching loss:
        L_flow = E[‖v_θ(a_t, t) - (a_1 - a_0)‖²]
    
    Args:
        v_pred:   (B, H, d_action) predicted velocity
        v_target: (B, H, d_action) target velocity = a_1 - a_0
    """
    return F.mse_loss(v_pred, v_target)


def silog_depth_loss(
    pred: torch.Tensor,   # (B, H, W) predicted depth
    gt: torch.Tensor,     # (B, H, W) GT depth
    mask: Optional[torch.Tensor] = None,  # (B, H, W) valid depth mask
    lambda_var: float = 0.85,
) -> torch.Tensor:
    """
    Scale-Invariant Log (SiLog) loss for depth supervision.
    
    SiLog(D̂, D_gt) = (1/n) Σ d_i² - (λ/n²)(Σ d_i)²
    where d_i = log(D̂_i) - log(D_gt_i)
    
    Used for supervising intermediate depth predictions.
    """
    eps = 1e-6
    if mask is None:
        mask = (gt > eps)

    pred_log = torch.log(pred.clamp(min=eps))
    gt_log   = torch.log(gt.clamp(min=eps))
    d = (pred_log - gt_log) * mask.float()

    n = mask.float().sum(dim=[1, 2]).clamp(min=1)
    loss_mse = (d ** 2).sum(dim=[1, 2]) / n
    loss_var = lambda_var * (d.sum(dim=[1, 2]) / n) ** 2
    return (loss_mse - loss_var).mean()


def gaussian_opacity_regularization(alpha: torch.Tensor, target_sparsity: float = 0.3) -> torch.Tensor:
    """
    Regularize Gaussian opacities towards a target sparsity.
    Encourages sparse meaningful 3D tokens.
    
    alpha: (B, N, 1) ∈ (0, 1)
    """
    mean_alpha = alpha.mean()
    return F.mse_loss(mean_alpha, torch.tensor(target_sparsity, device=alpha.device))


def gaussian_scale_regularization(log_scale: torch.Tensor, max_scale: float = 0.5) -> torch.Tensor:
    """
    Penalize excessively large Gaussian scales (prevents degenerate solutions).
    log_scale: (B, N, 3)
    """
    scale = torch.exp(log_scale)
    penalty = F.relu(scale - max_scale).mean()
    return penalty


class GSTVLALoss(nn.Module):
    """
    Combined loss for GST-VLA training.
    
    L_total = L_flow + λ₁*L_depth + λ₂*L_opacity + λ₃*L_scale
    
    For ACCV (no CoT):
        - L_flow:   flow matching (primary)
        - L_depth:  depth supervision (optional, for GST pretraining)
        - L_opacity: GST regularization
        - L_scale:   GST regularization
    """

    def __init__(
        self,
        lambda_depth:   float = 0.1,
        lambda_opacity: float = 0.01,
        lambda_scale:   float = 0.001,
        target_sparsity: float = 0.3,
    ):
        super().__init__()
        self.lambda_depth   = lambda_depth
        self.lambda_opacity = lambda_opacity
        self.lambda_scale   = lambda_scale
        self.target_sparsity = target_sparsity

    def forward(
        self,
        loss_flow: torch.Tensor,
        gst_aux: Dict[str, torch.Tensor],
        pred_depth: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        depth_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            loss_flow:  flow matching loss (scalar)
            gst_aux:    from GST forward (contains alpha, log_scale)
            pred_depth: (B, H, W) if depth supervision enabled
            gt_depth:   (B, H, W) ground truth depth
        
        Returns:
            dict with individual losses and total
        """
        losses = {"loss_flow": loss_flow}
        total = loss_flow

        # Depth supervision (when GT depth available)
        if pred_depth is not None and gt_depth is not None:
            loss_depth = silog_depth_loss(pred_depth, gt_depth, depth_mask)
            losses["loss_depth"] = loss_depth
            total = total + self.lambda_depth * loss_depth

        # GST regularization
        if "alpha" in gst_aux:
            loss_opacity = gaussian_opacity_regularization(
                gst_aux["alpha"], self.target_sparsity
            )
            losses["loss_opacity"] = loss_opacity
            total = total + self.lambda_opacity * loss_opacity

        if "log_scale" in gst_aux:
            loss_scale = gaussian_scale_regularization(gst_aux["log_scale"])
            losses["loss_scale"] = loss_scale
            total = total + self.lambda_scale * loss_scale

        losses["loss_total"] = total
        return losses


# ─────────────────────────────────────────────
# 3-Stage Trainer
# ─────────────────────────────────────────────

class GSTVLATrainer:
    """
    3-Stage training pipeline for GST-VLA.
    
    Stage 1: GST + projector pretraining (no VLM update)
    Stage 2: Add LoRA to VLM + flow expert warmup
    Stage 3: Full fine-tuning

    Usage:
        trainer = GSTVLATrainer(model, config)
        trainer.set_stage(1)
        trainer.train_epoch(dataloader)
        ...
        trainer.set_stage(2)
        ...
    """

    STAGE_CONFIGS = {
        1: {
            "description": "S1: GST + Expert (frozen VLM + SigLIP)",
            "trainable": ["gst", "vlm_projector", "action_expert"],
            "frozen":    ["dual_encoder", "vlm"],
            "lr":         1e-4,
        },
        2: {
            "description": "S2: +LoRA on VLM projector + flow expert",
            "trainable": ["gst", "vlm_projector", "action_expert"],  # + VLM LoRA
            "frozen":    ["dual_encoder"],
            "lr":         5e-5,
        },
        3: {
            "description": "S3: Full fine-tune",
            "trainable": ["gst", "vlm_projector", "action_expert"],
            "frozen":    ["dual_encoder"],
            "lr":         1e-5,
        },
    }

    def __init__(self, model, config: dict, device: torch.device):
        self.model  = model
        self.config = config
        self.device = device
        self.stage  = 1
        self.criterion = GSTVLALoss(
            lambda_depth=config.get("lambda_depth", 0.1),
            lambda_opacity=config.get("lambda_opacity", 0.01),
            lambda_scale=config.get("lambda_scale", 0.001),
        )
        self.optimizer = None
        self.scaler = torch.cuda.amp.GradScaler()  # mixed precision

    def set_stage(self, stage: int):
        """Configure trainable parameters for the given stage."""
        assert stage in [1, 2, 3], f"Stage must be 1, 2, or 3"
        self.stage = stage
        cfg = self.STAGE_CONFIGS[stage]
        print(f"\n[Trainer] {cfg['description']}")

        # Freeze all first
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Unfreeze trainable components
        for component_name in cfg["trainable"]:
            component = getattr(self.model, component_name, None)
            if component is not None:
                for p in component.parameters():
                    p.requires_grad_(True)
                n = sum(p.numel() for p in component.parameters() if p.requires_grad)
                print(f"  ✓ {component_name}: {n/1e6:.1f}M trainable params")

        # Build optimizer for trainable params only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        n_total = sum(p.numel() for p in trainable_params)
        print(f"  Total trainable: {n_total/1e6:.1f}M\n")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg["lr"],
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with mixed precision."""
        self.model.train()
        self.optimizer.zero_grad()

        # Move to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.cuda.amp.autocast():
            output = self.model(
                rgb=batch["rgb"],
                instruction_ids=batch["instruction_ids"],
                attention_mask=batch["attention_mask"],
                robot_state=batch["robot_state"],
                gt_actions=batch["actions"],
                camera_K=batch.get("camera_K"),
            )

            losses = self.criterion(
                loss_flow=output["loss_flow"],
                gst_aux=output["gst_aux"],
                pred_depth=output.get("pred_depth"),
                gt_depth=batch.get("depth_gt"),
            )

        self.scaler.scale(losses["loss_total"]).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step."""
        self.model.eval()
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        output = self.model(
            rgb=batch["rgb"],
            instruction_ids=batch["instruction_ids"],
            attention_mask=batch["attention_mask"],
            robot_state=batch["robot_state"],
            gt_actions=batch["actions"],
        )

        losses = self.criterion(
            loss_flow=output["loss_flow"],
            gst_aux=output["gst_aux"],
        )
        return {k: v.item() for k, v in losses.items()}
