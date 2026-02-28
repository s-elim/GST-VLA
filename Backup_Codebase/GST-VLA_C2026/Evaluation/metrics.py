"""
Evaluation Metrics for GST-VLA
================================
ACCV 2026

Metrics:
    1. Task Success Rate (SR)       — primary robot manipulation metric
    2. Action L2 Distance           — open-loop trajectory quality
    3. Action Smoothness            — joint jerk / oscillation measure
    4. 3D Spatial Token Quality     — GST-specific: depth alignment, coverage
    5. Depth Prediction Error       — AbsRel, SiLog, δ<1.25 thresholds
    6. Flow Matching Quality        — NFD (Normalized Flow Deviation)
    7. Inference Latency            — wall-clock timing (ms)

Usage:
    evaluator = GSTVLAEvaluator(model, device)
    results = evaluator.evaluate(dataloader)
    evaluator.print_report(results)
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# Metric Dataclass
# ─────────────────────────────────────────────

@dataclass
class EvalResults:
    """Container for all evaluation results."""

    # Task performance
    success_rate:       float = 0.0   # fraction of successful episodes
    partial_success:    float = 0.0   # partial task completion

    # Action quality (open-loop)
    action_l2_mean:     float = 0.0   # mean L2 between pred and GT actions
    action_l2_std:      float = 0.0
    action_smoothness:  float = 0.0   # lower = smoother

    # Depth metrics
    depth_absrel:       float = 0.0   # absolute relative error
    depth_silog:        float = 0.0   # scale-invariant log error
    depth_delta1:       float = 0.0   # % pixels with δ < 1.25
    depth_delta2:       float = 0.0   # % pixels with δ < 1.25²
    depth_delta3:       float = 0.0   # % pixels with δ < 1.25³

    # GST spatial token quality
    gst_depth_alignment: float = 0.0  # 3D token depth alignment error
    gst_coverage:        float = 0.0  # scene coverage (% of valid 3D space)
    gst_entropy:         float = 0.0  # token distribution entropy

    # Flow matching
    flow_nfd:            float = 0.0  # normalized flow deviation

    # Latency (ms)
    latency_mean_ms:    float = 0.0
    latency_p95_ms:     float = 0.0
    latency_p99_ms:     float = 0.0

    # Per-task breakdown
    per_task_sr:        Dict[str, float] = field(default_factory=dict)

    # Sample count
    n_samples:          int = 0
    n_episodes:         int = 0

    def to_dict(self) -> Dict[str, float]:
        d = {}
        for f_name in self.__dataclass_fields__:
            val = getattr(self, f_name)
            if isinstance(val, (int, float)):
                d[f_name] = val
        return d


# ─────────────────────────────────────────────
# Individual Metric Functions
# ─────────────────────────────────────────────

def action_l2_distance(
    pred: torch.Tensor,   # (B, H, d_action)
    gt:   torch.Tensor,   # (B, H, d_action)
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Per-step L2 distance between predicted and GT action chunks.
    Returns mean over horizon H.
    """
    diff = pred - gt                          # (B, H, d)
    l2   = diff.norm(dim=-1)                  # (B, H)
    l2_per_sample = l2.mean(dim=-1)           # (B,)  mean over horizon
    if reduction == "mean":
        return l2_per_sample.mean()
    return l2_per_sample


def action_smoothness(
    actions: torch.Tensor,  # (B, H, d_action) or (T, d_action)
) -> torch.Tensor:
    """
    Action smoothness: mean squared second-order finite difference (jerk proxy).
    
    smoothness = mean( ‖a_{t+1} - 2*a_t + a_{t-1}‖² )
    
    Lower = smoother trajectory.
    """
    if actions.dim() == 3:
        # (B, H, d): compute per batch
        d1 = actions[:, 1:, :] - actions[:, :-1, :]       # (B, H-1, d)
        d2 = d1[:, 1:, :] - d1[:, :-1, :]                 # (B, H-2, d)
        return (d2 ** 2).sum(dim=-1).mean()
    else:
        # (T, d)
        d1 = actions[1:] - actions[:-1]
        d2 = d1[1:] - d1[:-1]
        return (d2 ** 2).sum(dim=-1).mean()


def depth_metrics(
    pred: torch.Tensor,   # (B, H, W) predicted depth
    gt:   torch.Tensor,   # (B, H, W) GT depth
    mask: Optional[torch.Tensor] = None,  # (B, H, W) valid pixels
    max_depth: float = 10.0,
) -> Dict[str, float]:
    """
    Standard depth evaluation metrics:
    - AbsRel:  mean(|d - d*| / d*)
    - SiLog:   scale-invariant log RMSE
    - δ < 1.25^k thresholds
    """
    eps = 1e-6

    if mask is None:
        mask = (gt > eps) & (gt < max_depth)

    pred_m = pred[mask].clamp(eps, max_depth)
    gt_m   = gt[mask].clamp(eps, max_depth)

    # Median scaling (affine-invariant alignment before evaluation)
    scale = torch.median(gt_m) / (torch.median(pred_m) + eps)
    pred_m_scaled = (pred_m * scale).clamp(eps, max_depth)

    # AbsRel
    abs_rel = ((pred_m_scaled - gt_m).abs() / gt_m).mean().item()

    # SiLog
    log_diff = torch.log(pred_m_scaled + eps) - torch.log(gt_m + eps)
    silog = torch.sqrt((log_diff ** 2).mean() - 0.5 * (log_diff.mean() ** 2)).item()

    # δ thresholds
    ratio = torch.maximum(pred_m_scaled / gt_m, gt_m / pred_m_scaled)
    delta1 = (ratio < 1.25     ).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    return {
        "absrel": abs_rel,
        "silog":  silog,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


def gst_depth_alignment(
    mu_3d: torch.Tensor,       # (B, N, 3) Gaussian centers in camera frame
    depth_map: torch.Tensor,   # (B, H, W) GT depth map
    patch_uv: torch.Tensor,    # (B, N, 2) patch UV coords
    H_img: int, W_img: int,
) -> float:
    """
    Measures how well the GST 3D token Z-coordinates align
    with actual depth values at the corresponding image locations.
    
    Metric: mean |Z_gaussian - D_gt| / D_gt
    """
    B, N, _ = mu_3d.shape
    device = mu_3d.device

    # Z-depth of Gaussian centers
    z_gst = mu_3d[..., 2].clamp(min=0.01)  # (B, N)

    # Sample GT depth at patch locations
    grid = patch_uv.clone()
    grid[..., 0] = (grid[..., 0] / (W_img - 1)) * 2 - 1
    grid[..., 1] = (grid[..., 1] / (H_img - 1)) * 2 - 1
    grid = grid.unsqueeze(2)  # (B, N, 1, 2)

    d_gt = F.grid_sample(
        depth_map.unsqueeze(1), grid,
        mode="bilinear", align_corners=True, padding_mode="border"
    ).squeeze(1).squeeze(-1)  # (B, N)

    valid = d_gt > 0.01
    if valid.sum() == 0:
        return 0.0

    rel_err = ((z_gst - d_gt).abs() / (d_gt + 1e-6))[valid]
    return rel_err.mean().item()


def gst_token_coverage(
    mu_3d: torch.Tensor,   # (B, N, 3) Gaussian centers
    alpha: torch.Tensor,   # (B, N, 1) opacities
    alpha_thresh: float = 0.3,
    grid_res: int = 8,
) -> float:
    """
    Measures spatial coverage of GST tokens:
    fraction of scene voxels containing at least one active token.
    
    Simple 3D grid occupancy check.
    """
    B, N, _ = mu_3d.shape

    # Only count active tokens
    active_mask = (alpha.squeeze(-1) > alpha_thresh)  # (B, N)

    total_coverage = 0.0
    for b in range(B):
        pts = mu_3d[b][active_mask[b]]  # (n_active, 3)
        if len(pts) == 0:
            continue

        # Normalize to [0, 1]
        mins = pts.min(dim=0).values
        maxs = pts.max(dim=0).values
        span = (maxs - mins).clamp(min=1e-6)
        pts_norm = (pts - mins) / span

        # Bin into grid
        indices = (pts_norm * (grid_res - 1)).long().clamp(0, grid_res - 1)
        # Count unique voxels
        voxel_ids = (indices[:, 0] * grid_res * grid_res +
                     indices[:, 1] * grid_res +
                     indices[:, 2])
        n_unique = voxel_ids.unique().numel()
        total_coverage += n_unique / (grid_res ** 3)

    return total_coverage / B


def gst_token_entropy(alpha: torch.Tensor) -> float:
    """
    Shannon entropy of Gaussian token opacity distribution.
    High entropy = well-distributed attention across scene.
    
    alpha: (B, N, 1)
    """
    a = alpha.squeeze(-1)  # (B, N)
    # Normalize to probability distribution
    a_prob = a / (a.sum(dim=-1, keepdim=True) + 1e-6)
    entropy = -(a_prob * (a_prob + 1e-10).log()).sum(dim=-1)  # (B,)
    return entropy.mean().item()


def normalized_flow_deviation(
    v_pred: torch.Tensor,    # (B, H, d) predicted velocity
    v_target: torch.Tensor,  # (B, H, d) target velocity
) -> float:
    """
    Normalized Flow Deviation (NFD):
    mean cosine distance between predicted and target velocity vectors.
    
    NFD = 1 - cosine_similarity(v_pred, v_target)
    NFD=0 means perfect direction; NFD=2 means opposite direction.
    """
    v_pred_flat   = v_pred.reshape(-1, v_pred.shape[-1])
    v_target_flat = v_target.reshape(-1, v_target.shape[-1])
    cos_sim = F.cosine_similarity(v_pred_flat, v_target_flat, dim=-1)
    nfd = (1 - cos_sim).mean().item()
    return nfd


# ─────────────────────────────────────────────
# Main Evaluator Class
# ─────────────────────────────────────────────

class GSTVLAEvaluator:
    """
    Comprehensive evaluator for GST-VLA.

    Computes all metrics from a dataloader of held-out episodes.

    Usage:
        evaluator = GSTVLAEvaluator(model, device)
        results   = evaluator.evaluate(val_loader)
        evaluator.print_report(results)
        evaluator.save_report(results, "eval_results.json")
    """

    def __init__(
        self,
        model,
        device: torch.device,
        max_depth: float = 5.0,
        alpha_thresh: float = 0.3,
    ):
        self.model        = model
        self.device       = device
        self.max_depth    = max_depth
        self.alpha_thresh = alpha_thresh

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        n_batches: Optional[int] = None,
        compute_latency: bool = True,
    ) -> EvalResults:
        """
        Run full evaluation over dataloader.
        
        Args:
            dataloader:      yields batches with rgb, depth_gt, actions, ...
            n_batches:       limit evaluation to N batches (None = all)
            compute_latency: measure inference timing
        
        Returns:
            EvalResults with all metrics populated
        """
        self.model.eval()

        # Accumulators
        l2_all         = []
        smooth_all     = []
        depth_all      = defaultdict(list)
        gst_align_all  = []
        gst_cov_all    = []
        gst_ent_all    = []
        nfd_all        = []
        latencies_ms   = []

        n_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            if n_batches is not None and batch_idx >= n_batches:
                break

            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            B = batch["rgb"].shape[0]
            n_samples += B

            # ── Latency timing ────────────────────────────
            if compute_latency:
                t0 = time.perf_counter()

            # ── Model forward ─────────────────────────────
            pred_actions, info = self.model.predict_action(
                rgb=batch["rgb"],
                instruction_ids=batch["instruction_ids"],
                attention_mask=batch["attention_mask"],
                robot_state=batch["robot_state"],
            )

            if compute_latency:
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) / B * 1000.0)  # per sample ms

            gt_actions = batch["actions"]  # (B, H, 7)

            # ── Action L2 ─────────────────────────────────
            l2 = action_l2_distance(pred_actions, gt_actions, reduction="none")
            l2_all.extend(l2.cpu().numpy().tolist())

            # ── Smoothness ────────────────────────────────
            sm = action_smoothness(pred_actions)
            smooth_all.append(sm.item())

            # ── Depth metrics (if GT depth available) ─────
            if "depth_gt" in batch and batch["depth_gt"] is not None:
                pred_depth = info.get("depth_map")
                if pred_depth is not None:
                    d_metrics = depth_metrics(
                        pred_depth, batch["depth_gt"],
                        max_depth=self.max_depth,
                    )
                    for k, v in d_metrics.items():
                        depth_all[k].append(v)

            # ── GST Token Quality ─────────────────────────
            if "mu_3d" in info and "alpha" in info:
                mu_3d = info["mu_3d"]   # (B, N, 3)
                alpha = info["alpha"]   # (B, N, 1)

                # Coverage and entropy
                cov = gst_token_coverage(mu_3d, alpha, self.alpha_thresh)
                ent = gst_token_entropy(alpha)
                gst_cov_all.append(cov)
                gst_ent_all.append(ent)

                # Depth alignment (if depth available)
                if "depth_gt" in batch:
                    H_img = batch["rgb"].shape[2]
                    W_img = batch["rgb"].shape[3]
                    # Build patch UV for alignment metric
                    from models.gst import GaussianSpatialTokenizer
                    n_patches_side = H_img // 14  # patch_size=14
                    n = n_patches_side
                    ps = 14
                    import torch
                    uv = torch.stack(
                        torch.meshgrid(
                            torch.arange(n, device=self.device) * ps + ps // 2,
                            torch.arange(n, device=self.device) * ps + ps // 2,
                            indexing="ij",
                        ), dim=-1
                    ).reshape(1, -1, 2).expand(B, -1, -1).float()[..., [1, 0]]

                    align = gst_depth_alignment(mu_3d, batch["depth_gt"], uv, H_img, W_img)
                    gst_align_all.append(align)

            # ── Flow NFD ──────────────────────────────────
            # Recompute flow prediction for NFD metric
            a0 = torch.randn_like(gt_actions)
            t  = torch.full((B,), 0.5, device=self.device)
            t_exp = t.view(B, 1, 1)
            a_mid = t_exp * gt_actions + (1 - t_exp) * a0
            v_target = gt_actions - a0

            h_vlm = info.get("h_vlm_cached")  # may be None if not cached
            if h_vlm is None:
                # Skip NFD if h_vlm not cached (lightweight eval mode)
                pass
            else:
                v_pred = self.model.action_expert(a_mid, t, h_vlm, batch["robot_state"])
                nfd = normalized_flow_deviation(v_pred, v_target)
                nfd_all.append(nfd)

        # ── Aggregate ─────────────────────────────────────
        results = EvalResults()
        results.n_samples = n_samples

        results.action_l2_mean    = float(np.mean(l2_all))  if l2_all    else 0.0
        results.action_l2_std     = float(np.std(l2_all))   if l2_all    else 0.0
        results.action_smoothness = float(np.mean(smooth_all)) if smooth_all else 0.0

        if depth_all:
            results.depth_absrel = float(np.mean(depth_all["absrel"]))
            results.depth_silog  = float(np.mean(depth_all["silog"]))
            results.depth_delta1 = float(np.mean(depth_all["delta1"]))
            results.depth_delta2 = float(np.mean(depth_all["delta2"]))
            results.depth_delta3 = float(np.mean(depth_all["delta3"]))

        results.gst_coverage       = float(np.mean(gst_cov_all))  if gst_cov_all  else 0.0
        results.gst_entropy        = float(np.mean(gst_ent_all))   if gst_ent_all  else 0.0
        results.gst_depth_alignment = float(np.mean(gst_align_all)) if gst_align_all else 0.0

        results.flow_nfd = float(np.mean(nfd_all)) if nfd_all else 0.0

        if latencies_ms:
            results.latency_mean_ms = float(np.mean(latencies_ms))
            results.latency_p95_ms  = float(np.percentile(latencies_ms, 95))
            results.latency_p99_ms  = float(np.percentile(latencies_ms, 99))

        return results

    @torch.no_grad()
    def evaluate_episode(
        self,
        episode_rgb: List[torch.Tensor],        # list of (3, H, W) frames
        episode_depth: List[torch.Tensor],      # list of (H, W)
        instruction_ids: torch.Tensor,          # (L,)
        attention_mask: torch.Tensor,           # (L,)
        robot_states: List[torch.Tensor],       # list of (d_state,)
        gt_actions: List[torch.Tensor],         # list of (7,) per timestep
        success_fn=None,                        # callable(final_state) → bool
    ) -> Dict[str, float]:
        """
        Evaluate a single rollout episode step-by-step.
        Used for closed-loop evaluation.
        
        Returns per-episode metrics.
        """
        T = len(episode_rgb)
        pred_actions_all = []

        for t in range(T):
            rgb   = episode_rgb[t].unsqueeze(0).to(self.device)
            ids   = instruction_ids.unsqueeze(0).to(self.device)
            mask  = attention_mask.unsqueeze(0).to(self.device)
            state = robot_states[t].unsqueeze(0).to(self.device)

            pred_chunk, _ = self.model.predict_action(rgb, ids, mask, state)
            pred_actions_all.append(pred_chunk[0, 0].cpu())  # first action of chunk

        pred_actions = torch.stack(pred_actions_all, dim=0)  # (T, 7)
        gt_acts      = torch.stack(gt_actions, dim=0)        # (T, 7)

        l2 = (pred_actions - gt_acts).norm(dim=-1).mean().item()
        sm = action_smoothness(pred_actions).item()

        metrics = {
            "action_l2":    l2,
            "smoothness":   sm,
            "success":      success_fn(robot_states[-1]) if success_fn else -1.0,
            "n_steps":      T,
        }
        return metrics

    def print_report(self, results: EvalResults, title: str = "GST-VLA Evaluation Report"):
        """Pretty-print evaluation report to console."""
        w = 60
        print("\n" + "═" * w)
        print(f"  {title}")
        print("═" * w)
        print(f"  Samples evaluated: {results.n_samples}")
        print()

        print("  ── Task Performance ──────────────────────────────")
        print(f"  Success Rate:              {results.success_rate*100:6.2f}%")
        print()

        print("  ── Action Quality ────────────────────────────────")
        print(f"  Action L2 (mean ± std):    {results.action_l2_mean:.4f} ± {results.action_l2_std:.4f}")
        print(f"  Action Smoothness:         {results.action_smoothness:.6f}  (↓ better)")
        print(f"  Flow NFD:                  {results.flow_nfd:.4f}  (↓ better, 0=perfect)")
        print()

        print("  ── Depth Estimation ──────────────────────────────")
        print(f"  AbsRel:                    {results.depth_absrel:.4f}  (↓ better)")
        print(f"  SiLog:                     {results.depth_silog:.4f}  (↓ better)")
        print(f"  δ < 1.25   (δ₁):           {results.depth_delta1*100:6.2f}%  (↑ better)")
        print(f"  δ < 1.25²  (δ₂):           {results.depth_delta2*100:6.2f}%")
        print(f"  δ < 1.25³  (δ₃):           {results.depth_delta3*100:6.2f}%")
        print()

        print("  ── GST Token Quality ─────────────────────────────")
        print(f"  Depth Alignment Error:     {results.gst_depth_alignment:.4f}  (↓ better)")
        print(f"  Scene Coverage:            {results.gst_coverage*100:6.2f}%  (↑ better)")
        print(f"  Token Entropy:             {results.gst_entropy:.4f}  (↑ more distributed)")
        print()

        print("  ── Inference Latency ─────────────────────────────")
        print(f"  Mean:                      {results.latency_mean_ms:.1f} ms")
        print(f"  P95:                       {results.latency_p95_ms:.1f} ms")
        print(f"  P99:                       {results.latency_p99_ms:.1f} ms")
        print(f"  Effective Hz:              {1000/max(results.latency_mean_ms,1):.1f} Hz")
        print()

        if results.per_task_sr:
            print("  ── Per-Task Success Rate ─────────────────────────")
            for task, sr in sorted(results.per_task_sr.items(), key=lambda x: -x[1]):
                bar = "█" * int(sr * 20)
                print(f"  {task:<25} {sr*100:5.1f}%  {bar}")
            print()

        print("═" * w + "\n")

    def save_report(self, results: EvalResults, path: str):
        """Save results to JSON."""
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = results.to_dict()
        data["per_task_sr"] = results.per_task_sr
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [✓] Report saved: {path}")


# Type hint helper
from typing import List, Optional
