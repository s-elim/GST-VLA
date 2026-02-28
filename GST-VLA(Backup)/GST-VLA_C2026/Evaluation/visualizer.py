"""
3D Gaussian Token Visualization for GST-VLA
=============================================
ACCV 2026

Generates publication-quality figures for paper:
    Fig 1. Gaussian token 3D scatter plot (color=opacity)
    Fig 2. Depth map + token overlay comparison
    Fig 3. Token coverage heatmap (top-down BEV)
    Fig 4. Training loss curves (multi-stage)
    Fig 5. Action trajectory comparison (pred vs GT)
    Fig 6. Per-token alpha distribution

All outputs saved as PNG (300 DPI for paper submission).

Usage:
    viz = GSTVisualizer(output_dir="viz_outputs")
    viz.plot_gaussian_tokens(mu_3d, alpha, depth_map, rgb)
    viz.plot_training_curves(loss_log)
    viz.plot_action_comparison(pred_actions, gt_actions)
"""

import numpy as np
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings

# Graceful import of visualization libs
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not installed. Visualization disabled.")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─────────────────────────────────────────────
# Color palettes (matches paper dark theme)
# ─────────────────────────────────────────────

COLORS = {
    "gst":      "#A78BFA",   # Purple — GST tokens
    "depth":    "#22D3EE",   # Cyan   — Depth
    "vlm":      "#34D399",   # Green  — VLM
    "expert":   "#F97316",   # Orange — Action Expert
    "gt":       "#6EE7B7",   # Mint   — Ground truth
    "pred":     "#FCD34D",   # Yellow — Prediction
    "bg":       "#0F172A",   # Dark BG
    "grid":     "#1E293B",   # Grid lines
    "text":     "#E2E8F0",   # Text
}

def _set_dark_style():
    """Apply dark paper-ready matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["grid"],
        "axes.edgecolor":    COLORS["text"],
        "axes.labelcolor":   COLORS["text"],
        "xtick.color":       COLORS["text"],
        "ytick.color":       COLORS["text"],
        "text.color":        COLORS["text"],
        "grid.color":        "#334155",
        "grid.alpha":        0.5,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": COLORS["bg"],
    })


def _tensor_to_numpy(x):
    """Safely convert tensor or ndarray to numpy."""
    if HAS_TORCH and isinstance(x, __import__("torch").Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


# ─────────────────────────────────────────────
# Main Visualizer
# ─────────────────────────────────────────────

class GSTVisualizer:
    """
    Publication-quality visualizer for GST-VLA analysis.
    """

    def __init__(self, output_dir: str = "viz_outputs", dark_mode: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dark_mode  = dark_mode
        if dark_mode and HAS_MPL:
            _set_dark_style()

    # ─────────────────────────────────────────
    # Fig 1: 3D Gaussian Token Scatter
    # ─────────────────────────────────────────

    def plot_gaussian_tokens_3d(
        self,
        mu_3d:  "ArrayLike",  # (N, 3) or (B, N, 3) — Gaussian centers
        alpha:  "ArrayLike",  # (N, 1) or (B, N, 1) — opacities
        title:  str = "GST: 3D Gaussian Token Distribution",
        save_path: Optional[str] = None,
        batch_idx: int = 0,
    ) -> str:
        """
        3D scatter plot of Gaussian token centers, colored by opacity.
        Larger + brighter = higher opacity (more important token).
        """
        if not HAS_MPL:
            return ""

        mu_np  = _tensor_to_numpy(mu_3d)
        alp_np = _tensor_to_numpy(alpha)

        # Handle batch dim
        if mu_np.ndim == 3:
            mu_np  = mu_np[batch_idx]
            alp_np = alp_np[batch_idx]
        alp_np = alp_np.squeeze(-1)  # (N,)

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(COLORS["grid"])

        # Normalize opacity for marker size + color
        alp_norm = (alp_np - alp_np.min()) / (alp_np.ptp() + 1e-6)
        sizes  = 10 + alp_norm * 80           # marker size
        colors = plt.cm.plasma(alp_norm)       # color by opacity

        sc = ax.scatter(
            mu_np[:, 0], mu_np[:, 2], mu_np[:, 1],  # X, Z, Y (Z=depth forward)
            c=alp_norm,
            s=sizes,
            cmap="plasma",
            alpha=0.85,
            edgecolors="none",
        )

        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("Token Opacity (α)", color=COLORS["text"])
        cbar.ax.yaxis.set_tick_params(color=COLORS["text"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["text"])

        ax.set_xlabel("X (m)", labelpad=8)
        ax.set_ylabel("Z / Depth (m)", labelpad=8)
        ax.set_zlabel("Y (m)", labelpad=8)
        ax.set_title(title)

        # Stats annotation
        n_active = (alp_np > 0.3).sum()
        ax.text2D(
            0.02, 0.95,
            f"N_tokens={len(mu_np)}  |  Active(α>0.3)={n_active}\n"
            f"Depth range: [{mu_np[:,2].min():.2f}, {mu_np[:,2].max():.2f}] m",
            transform=ax.transAxes,
            fontsize=9,
            color=COLORS["text"],
            bbox=dict(boxstyle="round", facecolor=COLORS["bg"], alpha=0.7),
        )

        save_path = save_path or str(self.output_dir / "fig1_gaussian_tokens_3d.png")
        plt.savefig(save_path)
        plt.close(fig)
        return save_path

    # ─────────────────────────────────────────
    # Fig 2: Depth Map + Token Overlay
    # ─────────────────────────────────────────

    def plot_depth_token_overlay(
        self,
        rgb:       "ArrayLike",        # (3, H, W) or (H, W, 3) normalized
        depth_map: "ArrayLike",        # (H, W)
        mu_3d:     "ArrayLike",        # (N, 3) or (B, N, 3)
        alpha:     "ArrayLike",        # (N, 1) or (B, N, 1)
        patch_uv:  "ArrayLike",        # (N, 2) or (B, N, 2) patch pixel coords
        save_path: Optional[str] = None,
        batch_idx: int = 0,
    ) -> str:
        """
        Side-by-side: RGB | Depth map | Token overlay on depth.
        Shows which image regions correspond to active GST tokens.
        """
        if not HAS_MPL:
            return ""

        rgb_np   = _tensor_to_numpy(rgb)
        dep_np   = _tensor_to_numpy(depth_map)
        mu_np    = _tensor_to_numpy(mu_3d)
        alp_np   = _tensor_to_numpy(alpha)
        uv_np    = _tensor_to_numpy(patch_uv)

        # Handle batch dim
        if rgb_np.ndim == 4:   rgb_np = rgb_np[batch_idx]
        if dep_np.ndim == 3:   dep_np = dep_np[batch_idx]
        if mu_np.ndim  == 3:   mu_np  = mu_np[batch_idx]
        if alp_np.ndim == 3:   alp_np = alp_np[batch_idx]
        if uv_np.ndim  == 3:   uv_np  = uv_np[batch_idx]
        alp_np = alp_np.squeeze(-1)

        # Denormalize RGB for display
        if rgb_np.shape[0] == 3:
            rgb_np = rgb_np.transpose(1, 2, 0)  # (H, W, 3)
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        rgb_disp = np.clip(rgb_np * std + mean, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("GST-VLA: Depth Map & Token Overlay", fontsize=14, y=1.01)

        # Panel 1: RGB
        axes[0].imshow(rgb_disp)
        axes[0].set_title("RGB Input")
        axes[0].axis("off")

        # Panel 2: Depth map
        im = axes[1].imshow(dep_np, cmap="plasma_r", interpolation="bilinear")
        axes[1].set_title("Metric Depth (Depth Anything V2)")
        axes[1].axis("off")
        cbar = plt.colorbar(im, ax=axes[1], shrink=0.85, pad=0.02)
        cbar.set_label("Depth (m)", color=COLORS["text"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["text"])

        # Panel 3: Token overlay on depth
        axes[2].imshow(dep_np, cmap="plasma_r", interpolation="bilinear", alpha=0.6)

        # Scatter active tokens at their patch UV locations
        alp_norm = (alp_np - alp_np.min()) / (alp_np.ptp() + 1e-6)
        active   = alp_norm > 0.15

        sc = axes[2].scatter(
            uv_np[active, 0],   # u = col
            uv_np[active, 1],   # v = row
            c=alp_norm[active],
            s=15 + alp_norm[active] * 60,
            cmap="YlOrRd",
            vmin=0, vmax=1,
            edgecolors="white",
            linewidths=0.3,
            alpha=0.9,
            zorder=5,
        )
        plt.colorbar(sc, ax=axes[2], shrink=0.85, pad=0.02).set_label(
            "Token α", color=COLORS["text"]
        )
        axes[2].set_title(f"GST Tokens Overlay  (N={active.sum()} active)")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = save_path or str(self.output_dir / "fig2_depth_token_overlay.png")
        plt.savefig(save_path)
        plt.close(fig)
        return save_path

    # ─────────────────────────────────────────
    # Fig 3: Bird's Eye View (BEV) Coverage Map
    # ─────────────────────────────────────────

    def plot_bev_coverage(
        self,
        mu_3d:    "ArrayLike",   # (N, 3) or (B, N, 3)
        alpha:    "ArrayLike",   # (N, 1) or (B, N, 1)
        grid_res: int = 64,
        save_path: Optional[str] = None,
        batch_idx: int = 0,
    ) -> str:
        """
        Top-down Bird's Eye View of Gaussian token coverage.
        Projects tokens to XZ plane (Z=depth), weighted by alpha.
        """
        if not HAS_MPL:
            return ""

        mu_np  = _tensor_to_numpy(mu_3d)
        alp_np = _tensor_to_numpy(alpha)
        if mu_np.ndim == 3:
            mu_np  = mu_np[batch_idx]
            alp_np = alp_np[batch_idx]
        alp_np = alp_np.squeeze(-1)

        # Build 2D occupancy map in XZ
        x = mu_np[:, 0]
        z = mu_np[:, 2]
        bev_grid = np.zeros((grid_res, grid_res))

        x_min, x_max = x.min(), x.max()
        z_min, z_max = z.min(), z.max()
        eps = 1e-6

        x_idx = ((x - x_min) / (x_max - x_min + eps) * (grid_res - 1)).astype(int).clip(0, grid_res - 1)
        z_idx = ((z - z_min) / (z_max - z_min + eps) * (grid_res - 1)).astype(int).clip(0, grid_res - 1)

        for i in range(len(mu_np)):
            bev_grid[z_idx[i], x_idx[i]] += alp_np[i]

        # Smooth
        from scipy.ndimage import gaussian_filter
        try:
            bev_smooth = gaussian_filter(bev_grid, sigma=1.5)
        except Exception:
            bev_smooth = bev_grid

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("GST Token Coverage — Bird's Eye View (Top-Down)", fontsize=13)

        # Left: BEV heatmap
        im = axes[0].imshow(
            bev_smooth,
            origin="lower",
            cmap="hot",
            interpolation="bilinear",
            extent=[x_min, x_max, z_min, z_max],
        )
        plt.colorbar(im, ax=axes[0]).set_label("Cumulative α", color=COLORS["text"])
        axes[0].set_xlabel("X (m)")
        axes[0].set_ylabel("Z / Depth (m)")
        axes[0].set_title("Alpha-Weighted Token Density")

        # Right: Raw scatter
        scatter_colors = plt.cm.plasma((alp_np - alp_np.min()) / (alp_np.ptp() + 1e-6))
        axes[1].scatter(x, z, c=scatter_colors, s=8, alpha=0.7)
        axes[1].set_xlabel("X (m)")
        axes[1].set_ylabel("Z / Depth (m)")
        axes[1].set_title(f"Token Scatter  (N_g={len(mu_np)})")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = save_path or str(self.output_dir / "fig3_bev_coverage.png")
        plt.savefig(save_path)
        plt.close(fig)
        return save_path

    # ─────────────────────────────────────────
    # Fig 4: Training Loss Curves (3-Stage)
    # ─────────────────────────────────────────

    def plot_training_curves(
        self,
        loss_log: Dict[str, List[float]],
        stage_boundaries: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Multi-panel training loss curves for 3-stage training.
        
        loss_log keys: 'train/loss_total', 'val/loss_total',
                       'train/loss_flow', 'train/loss_depth', etc.
        stage_boundaries: list of step indices where stage transitions occur.
        """
        if not HAS_MPL:
            return ""

        fig = plt.figure(figsize=(16, 10))
        gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        ax_total  = fig.add_subplot(gs[0, :2])
        ax_flow   = fig.add_subplot(gs[0, 2])
        ax_depth  = fig.add_subplot(gs[1, 0])
        ax_opac   = fig.add_subplot(gs[1, 1])
        ax_lr     = fig.add_subplot(gs[1, 2])

        fig.suptitle("GST-VLA Training Curves — 3-Stage Pipeline", fontsize=14, y=1.01)

        def _plot(ax, key_train, key_val=None, title="", ylabel="Loss", color=None):
            color = color or COLORS["gst"]
            if key_train in loss_log:
                vals = loss_log[key_train]
                ax.plot(vals, color=color, linewidth=1.5, label="Train", alpha=0.9)
                # Smoothed line
                if len(vals) > 20:
                    kernel = np.ones(20) / 20
                    smooth = np.convolve(vals, kernel, mode="valid")
                    offset = len(vals) - len(smooth)
                    ax.plot(
                        range(offset // 2, offset // 2 + len(smooth)),
                        smooth, color=color, linewidth=2.5, alpha=1.0,
                    )

            if key_val and key_val in loss_log:
                ax.plot(loss_log[key_val], color=COLORS["gt"],
                        linewidth=1.5, linestyle="--", label="Val", alpha=0.9)

            # Stage boundaries
            if stage_boundaries:
                for i, s in enumerate(stage_boundaries):
                    ax.axvline(s, color="#F59E0B", linewidth=1.2,
                               linestyle=":", alpha=0.8)
                    ax.text(s, ax.get_ylim()[1] * 0.9, f"S{i+2}",
                            color="#F59E0B", fontsize=8, ha="center")

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Steps")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        _plot(ax_total, "train/loss_total", "val/loss_total",
              title="Total Loss", color=COLORS["gst"])
        _plot(ax_flow,  "train/loss_flow",  "val/loss_flow",
              title="Flow Matching Loss L_flow", color=COLORS["expert"])
        _plot(ax_depth, "train/loss_depth",
              title="Depth Loss L_depth", color=COLORS["depth"])
        _plot(ax_opac,  "train/loss_opacity",
              title="Opacity Regularization", color=COLORS["vlm"])
        _plot(ax_lr,    "lr",
              title="Learning Rate", ylabel="LR", color="#F59E0B")

        save_path = save_path or str(self.output_dir / "fig4_training_curves.png")
        plt.savefig(save_path)
        plt.close(fig)
        return save_path

    # ─────────────────────────────────────────
    # Fig 5: Action Trajectory Comparison
    # ─────────────────────────────────────────

    def plot_action_comparison(
        self,
        pred_actions: "ArrayLike",   # (H, 7) or (B, H, 7)
        gt_actions:   "ArrayLike",   # (H, 7) or (B, H, 7)
        action_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        batch_idx: int = 0,
    ) -> str:
        """
        Per-DoF action trajectory comparison: predicted vs ground truth.
        Shows all 7 action dimensions: Δpose(6) + gripper(1).
        """
        if not HAS_MPL:
            return ""

        pred_np = _tensor_to_numpy(pred_actions)
        gt_np   = _tensor_to_numpy(gt_actions)

        if pred_np.ndim == 3:
            pred_np = pred_np[batch_idx]
            gt_np   = gt_np[batch_idx]

        H, D = pred_np.shape
        action_names = action_names or [
            "Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"
        ]
        colors_pred = ["#FCD34D"] * 6 + ["#F97316"]
        colors_gt   = ["#6EE7B7"] * 6 + ["#34D399"]

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes_flat = axes.flatten()
        fig.suptitle("GST-VLA: Predicted vs GT Action Trajectory (H=16 chunk)", fontsize=13)

        steps = np.arange(H)
        for i in range(D):
            ax = axes_flat[i]
            ax.plot(steps, gt_np[:, i],   color=colors_gt[i],   linewidth=2.0,
                    label="GT", alpha=0.9, linestyle="--")
            ax.plot(steps, pred_np[:, i], color=colors_pred[i], linewidth=2.0,
                    label="Pred", alpha=0.9)

            l2 = np.sqrt(((pred_np[:, i] - gt_np[:, i]) ** 2).mean())
            ax.set_title(f"{action_names[i]}  (L2={l2:.4f})")
            ax.set_xlabel("Horizon step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Summary panel (last subplot)
        ax_sum = axes_flat[D]
        l2_per_dim = np.sqrt(((pred_np - gt_np) ** 2).mean(axis=0))
        bars = ax_sum.bar(action_names, l2_per_dim,
                          color=colors_pred, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax_sum.set_title("L2 Error per Action Dimension")
        ax_sum.set_ylabel("RMSE")
        ax_sum.tick_params(axis="x", rotation=30)
        ax_sum.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, l2_per_dim):
            ax_sum.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        save_path = save_path or str(self.output_dir / "fig5_action_comparison.png")
        plt.savefig(save_path)
        plt.close(fig)
        return save_path

    # ─────────────────────────────────────────
    # Fig 6: Alpha Distribution Histogram
    # ─────────────────────────────────────────

    def plot_alpha_distribution(
        self,
        alpha_list: List["ArrayLike"],   # list of (N, 1) from multiple samples
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Distribution of Gaussian token opacities across multiple samples.
        Shows whether tokens are well-distributed or collapsed.
        """
        if not HAS_MPL:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("GST: Gaussian Token Opacity (α) Distribution", fontsize=13)

        palette = plt.cm.plasma(np.linspace(0.2, 0.9, len(alpha_list)))

        all_alphas = []
        for i, (alp, color) in enumerate(zip(alpha_list, palette)):
            alp_np = _tensor_to_numpy(alp).squeeze()
            if alp_np.ndim > 1:
                alp_np = alp_np.flatten()
            label = labels[i] if labels else f"Sample {i}"
            axes[0].hist(alp_np, bins=50, alpha=0.5, color=color, label=label, density=True)
            all_alphas.append(alp_np)

        axes[0].axvline(0.3, color="#F59E0B", linestyle="--", linewidth=1.5, label="thresh=0.3")
        axes[0].set_xlabel("Token Opacity α")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Per-Sample Alpha Distribution")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # CDF
        all_np = np.concatenate(all_alphas)
        sorted_alpha = np.sort(all_np)
        cdf = np.arange(len(sorted_alpha)) / len(sorted_alpha)
        axes[1].plot(sorted_alpha, cdf, color=COLORS["gst"], linewidth=2)
        axes[1].axvline(0.3, color="#F59E0B", linestyle="--", linewidth=1.5,
                        label=f"thresh=0.3 → {(all_np>0.3).mean()*100:.1f}% active")
        axes[1].set_xlabel("Token Opacity α")
        axes[1].set_ylabel("CDF")
        axes[1].set_title("Cumulative Distribution")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = save_path or str(self.output_dir / "fig6_alpha_distribution.png")
        plt.savefig(save_path)
        plt.close(fig)
        return save_path

    # ─────────────────────────────────────────
    # Composite: Full Eval Summary Figure
    # ─────────────────────────────────────────

    def plot_eval_summary(
        self,
        results_dict: Dict[str, "EvalResults"],  # model_name → EvalResults
        save_path: Optional[str] = None,
    ) -> str:
        """
        Comparison bar chart across models (for paper Table / Figure).
        
        results_dict: {"GST-VLA (Ours)": results, "Baseline": results, ...}
        """
        if not HAS_MPL:
            return ""

        metrics_to_plot = [
            ("action_l2_mean",      "Action L2 ↓",     True),
            ("action_smoothness",   "Smoothness ↓",    True),
            ("depth_delta1",        "Depth δ₁ ↑",      False),
            ("depth_absrel",        "Depth AbsRel ↓",  True),
            ("gst_coverage",        "Token Coverage ↑",False),
            ("gst_entropy",         "Token Entropy ↑", False),
        ]

        n_metrics = len(metrics_to_plot)
        n_models  = len(results_dict)
        names     = list(results_dict.keys())
        palette   = plt.cm.tab10(np.linspace(0, 0.8, n_models))

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes_flat = axes.flatten()
        fig.suptitle("GST-VLA Model Comparison", fontsize=14)

        for i, (metric_key, label, lower_better) in enumerate(metrics_to_plot):
            ax = axes_flat[i]
            vals = [getattr(results_dict[name], metric_key, 0.0) for name in names]

            bars = ax.bar(
                range(n_models), vals,
                color=palette, alpha=0.85,
                edgecolor="white", linewidth=0.7,
            )
            ax.set_xticks(range(n_models))
            ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
            ax.set_title(label)
            ax.grid(True, alpha=0.3, axis="y")

            # Highlight best
            best_idx = int(np.argmin(vals) if lower_better else np.argmax(vals))
            bars[best_idx].set_edgecolor("#FCD34D")
            bars[best_idx].set_linewidth(2.5)

            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                )

        plt.tight_layout()
        save_path = save_path or str(self.output_dir / "fig_eval_summary.png")
        plt.savefig(save_path)
        plt.close(fig)
        return save_path


# ─────────────────────────────────────────────
# Standalone helper: generate all viz from checkpoint
# ─────────────────────────────────────────────

def generate_paper_figures(
    model,
    dataloader,
    device,
    output_dir: str = "paper_figures",
    n_samples: int = 4,
):
    """
    One-shot: run model on N samples and generate all paper-ready figures.
    
    Usage:
        generate_paper_figures(model, val_loader, device, "figures/")
    """
    import torch

    viz     = GSTVisualizer(output_dir=output_dir, dark_mode=True)
    model.eval()
    saved   = []

    alpha_samples = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            pred_actions, info = model.predict_action(
                rgb=batch["rgb"],
                instruction_ids=batch["instruction_ids"],
                attention_mask=batch["attention_mask"],
                robot_state=batch["robot_state"],
            )

            mu_3d     = info["mu_3d"]
            alpha     = info["alpha"]
            depth_map = info["depth_map"]

            # Get patch UV (reproduce from GST config)
            n_side = 224 // 14
            ps     = 14
            coords = torch.stack(
                torch.meshgrid(
                    torch.arange(n_side, device=device) * ps + ps // 2,
                    torch.arange(n_side, device=device) * ps + ps // 2,
                    indexing="ij",
                ), dim=-1
            ).reshape(1, -1, 2).expand(batch["rgb"].shape[0], -1, -1).float()[..., [1, 0]]

            alpha_samples.append(alpha[0].cpu())

            p = viz.plot_gaussian_tokens_3d(mu_3d[0], alpha[0],
                    title=f"GST Tokens — Sample {i+1}",
                    save_path=str(Path(output_dir) / f"sample{i+1}_fig1_tokens3d.png"))
            saved.append(p)

            p = viz.plot_depth_token_overlay(batch["rgb"][0], depth_map[0],
                    mu_3d[0], alpha[0], coords[0],
                    save_path=str(Path(output_dir) / f"sample{i+1}_fig2_overlay.png"))
            saved.append(p)

            p = viz.plot_bev_coverage(mu_3d[0], alpha[0],
                    save_path=str(Path(output_dir) / f"sample{i+1}_fig3_bev.png"))
            saved.append(p)

            if "actions" in batch:
                p = viz.plot_action_comparison(pred_actions[0], batch["actions"][0],
                        save_path=str(Path(output_dir) / f"sample{i+1}_fig5_actions.png"))
                saved.append(p)

    # Global alpha distribution
    p = viz.plot_alpha_distribution(alpha_samples,
            labels=[f"Sample {i+1}" for i in range(len(alpha_samples))],
            save_path=str(Path(output_dir) / "fig6_alpha_dist.png"))
    saved.append(p)

    print(f"\n[Visualizer] Generated {len(saved)} figures → {output_dir}/")
    for s in saved:
        if s:
            print(f"  ✓ {s}")
    return saved
