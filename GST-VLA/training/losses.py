"""
Training Losses & 3-Stage Trainer for DEAD-VLA
===============================================
ACCV 2026

Loss functions:
    L_flow  = E[‖v_θ − (a_1 − a_0)‖²]        (Conditional Flow Matching)
    L_CoT   = mean SmoothL1(pred, gt)           (DA-CoT supervision, optional)
    L_depth = SiLog(D̂, D_gt)                   (Depth supervision, optional)
    L_total = L_flow + λ_CoT*L_CoT + λ_depth*L_depth + λ_opacity*L_opacity + λ_scale*L_scale

3-Stage Training:
    S1: GST + State Encoder + DA-CoT + Action Expert  (frozen VLM + encoders)
    S2: +LoRA on VLM  (gradients flow through projector into LoRA layers)
    S3: Full fine-tune (all trainable components)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ─────────────────────────────────────────────
# Individual Loss Functions
# ─────────────────────────────────────────────

def flow_matching_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """L_flow = E[‖v_θ(a_t, t) − (a_1 − a_0)‖²]"""
    return F.mse_loss(v_pred, v_target)


def cot_loss(
    reasoning: Dict[str, torch.Tensor],
    cot_targets: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    L_CoT: DA-CoT supervision loss.

    When cot_targets is None (unsupervised setting), returns 0.
    Otherwise computes SmoothL1 over each available annotation type.
    """
    device = next(iter(reasoning.values())).device

    if cot_targets is None:
        return torch.zeros(1, device=device).squeeze()

    losses = []
    for key in ["obj_grounding", "grasp_affordance", "spatial_relations", "motion_plan"]:
        if key in cot_targets and cot_targets[key] is not None:
            pred = reasoning[key]
            gt   = cot_targets[key].to(device)
            min_k = min(pred.shape[1], gt.shape[1])
            losses.append(F.smooth_l1_loss(pred[:, :min_k], gt[:, :min_k]))

    if not losses:
        return torch.zeros(1, device=device).squeeze()
    return torch.stack(losses).mean()


def silog_depth_loss(
    pred: torch.Tensor,               # (B, H, W) predicted depth
    gt: torch.Tensor,                 # (B, H, W) GT depth
    mask: Optional[torch.Tensor] = None,
    lambda_var: float = 0.85,
) -> torch.Tensor:
    """
    Scale-Invariant Log loss for depth supervision.
    SiLog(D̂, D_gt) = (1/n) Σ d_i² − (λ/n²)(Σ d_i)²
    where d_i = log(D̂_i) − log(D_gt_i)
    """
    eps = 1e-6
    if mask is None:
        mask = (gt > eps)
    d = (torch.log(pred.clamp(min=eps)) - torch.log(gt.clamp(min=eps))) * mask.float()
    n = mask.float().sum(dim=[1, 2]).clamp(min=1)
    loss_mse = (d ** 2).sum(dim=[1, 2]) / n
    loss_var = lambda_var * (d.sum(dim=[1, 2]) / n) ** 2
    return (loss_mse - loss_var).mean()


def gaussian_opacity_regularization(alpha: torch.Tensor, target_sparsity: float = 0.3) -> torch.Tensor:
    """Encourage Gaussian opacities toward target_sparsity. alpha: (B, N, 1)."""
    return F.mse_loss(alpha.mean(), torch.tensor(target_sparsity, device=alpha.device))


def gaussian_scale_regularization(log_scale: torch.Tensor, max_scale: float = 0.5) -> torch.Tensor:
    """Penalize excessively large Gaussian scales. log_scale: (B, N, 3)."""
    return F.relu(torch.exp(log_scale) - max_scale).mean()


# ─────────────────────────────────────────────
# Combined Loss Module
# ─────────────────────────────────────────────

class GSTVLALoss(nn.Module):
    """
    Combined training loss for DEAD-VLA.

    L_total = L_flow
            + λ_CoT   * L_CoT
            + λ_depth * L_depth     (when GT depth available)
            + λ_opacity * L_opacity  (GST regularization)
            + λ_scale   * L_scale    (GST regularization)
    """

    def __init__(
        self,
        lambda_cot: float = 0.1,
        lambda_depth: float = 0.1,
        lambda_opacity: float = 0.01,
        lambda_scale: float = 0.001,
        target_sparsity: float = 0.3,
    ):
        super().__init__()
        self.lambda_cot     = lambda_cot
        self.lambda_depth   = lambda_depth
        self.lambda_opacity = lambda_opacity
        self.lambda_scale   = lambda_scale
        self.target_sparsity = target_sparsity

    def forward(
        self,
        loss_flow: torch.Tensor,
        gst_aux: Dict[str, torch.Tensor],
        loss_cot: Optional[torch.Tensor] = None,
        pred_depth: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        depth_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            loss_flow:  flow matching loss (scalar)
            gst_aux:    from GST forward (contains alpha, log_scale)
            loss_cot:   DA-CoT loss scalar (or None)
            pred_depth: (B, H, W) if depth supervision enabled
            gt_depth:   (B, H, W) ground truth metric depth
        """
        losses = {"loss_flow": loss_flow}
        total  = loss_flow

        # DA-CoT loss
        if loss_cot is not None:
            losses["loss_cot"] = loss_cot
            total = total + self.lambda_cot * loss_cot

        # Depth supervision
        if pred_depth is not None and gt_depth is not None:
            ld = silog_depth_loss(pred_depth, gt_depth, depth_mask)
            losses["loss_depth"] = ld
            total = total + self.lambda_depth * ld

        # GST regularization
        if "alpha" in gst_aux:
            lo = gaussian_opacity_regularization(gst_aux["alpha"], self.target_sparsity)
            losses["loss_opacity"] = lo
            total = total + self.lambda_opacity * lo

        if "log_scale" in gst_aux:
            ls = gaussian_scale_regularization(gst_aux["log_scale"])
            losses["loss_scale"] = ls
            total = total + self.lambda_scale * ls

        losses["loss_total"] = total
        return losses


# ─────────────────────────────────────────────
# 3-Stage Trainer
# ─────────────────────────────────────────────

class GSTVLATrainer:
    """
    3-Stage training pipeline for DEAD-VLA.

    Stage 1: GST + state encoder + DA-CoT + action expert
             (VLM, SigLIP, DepthV2 frozen)
    Stage 2: + LoRA on VLM cross-attention
    Stage 3: Full fine-tune

    Usage:
        trainer = GSTVLATrainer(model, config, device)
        trainer.set_stage(1)
        for batch in train_loader:
            metrics = trainer.train_step(batch)
    """

    STAGE_CONFIGS = {
        1: {
            "description": "S1: GST + State Encoder + DA-CoT + Expert (frozen VLM)",
            "trainable": ["gst", "vlm_projector", "state_encoder", "da_cot", "action_expert"],
            "frozen":    ["dual_encoder", "vlm"],
            "lr":         1e-4,
        },
        2: {
            "description": "S2: +LoRA on VLM",
            "trainable": ["gst", "vlm_projector", "state_encoder", "da_cot", "action_expert"],
            "frozen":    ["dual_encoder"],
            "lr":         5e-5,
        },
        3: {
            "description": "S3: Full fine-tune",
            "trainable": ["gst", "vlm_projector", "state_encoder", "da_cot", "action_expert"],
            "frozen":    ["dual_encoder"],
            "lr":         1e-5,
        },
    }

    def __init__(self, model, config: dict, device: torch.device):
        self.model     = model
        self.config    = config
        self.device    = device
        self.stage     = 1
        self.criterion = GSTVLALoss(
            lambda_cot=config.get("lambda_cot", 0.1),
            lambda_depth=config.get("lambda_depth", 0.1),
            lambda_opacity=config.get("lambda_opacity", 0.01),
            lambda_scale=config.get("lambda_scale", 0.001),
        )
        self.optimizer = None
        self.scaler    = torch.cuda.amp.GradScaler()

    def set_stage(self, stage: int):
        """Configure trainable parameters for the given training stage."""
        assert stage in [1, 2, 3]
        self.stage = stage
        cfg = self.STAGE_CONFIGS[stage]
        print(f"\n[Trainer] {cfg['description']}")

        for p in self.model.parameters():
            p.requires_grad_(False)

        for name in cfg["trainable"]:
            component = getattr(self.model, name, None)
            if component is not None:
                for p in component.parameters():
                    p.requires_grad_(True)
                n = sum(p.numel() for p in component.parameters() if p.requires_grad)
                print(f"  ✓ {name}: {n/1e6:.2f}M trainable")

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        n_total = sum(p.numel() for p in trainable)
        print(f"  Total trainable: {n_total/1e6:.2f}M\n")

        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg["lr"], weight_decay=0.01, betas=(0.9, 0.95),
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with mixed-precision (bf16)."""
        self.model.train()
        self.optimizer.zero_grad()

        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self.model(
                rgb=batch["rgb"],
                instruction_ids=batch["instruction_ids"],
                attention_mask=batch["attention_mask"],
                robot_state=batch["robot_state"],
                gt_actions=batch["actions"],
                camera_K=batch.get("camera_K"),
                cot_targets=batch.get("cot_targets"),
            )

            losses = self.criterion(
                loss_flow=output["loss_flow"],
                gst_aux=output["gst_aux"],
                loss_cot=output.get("loss_cot"),
                pred_depth=output.get("pred_depth"),
                gt_depth=batch.get("depth_gt"),
            )

        self.scaler.scale(losses["loss_total"]).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single validation step."""
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
            loss_cot=output.get("loss_cot"),
        )
        return {k: v.item() for k, v in losses.items()}
