"""
GST-VLA Evaluation Script
===========================
ACCV 2026

Runs full evaluation suite:
  1. Load model from checkpoint
  2. Run metrics on validation set
  3. Generate paper-ready visualizations
  4. Save JSON report

Usage:
    # Evaluate with mock data (no checkpoint needed)
    python evaluate.py --mock

    # Evaluate from checkpoint
    python evaluate.py --checkpoint checkpoints/stage3/best.pt --data_root /data

    # Evaluate + generate paper figures
    python evaluate.py --checkpoint checkpoints/stage3/best.pt --data_root /data --visualize

    # Compare multiple checkpoints
    python evaluate.py --compare stage1=ckpt/s1.pt stage3=ckpt/s3.pt --mock
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.gst_vla import GSTVLA
from evaluation.metrics import GSTVLAEvaluator, EvalResults
from visualization.visualizer import GSTVisualizer, generate_paper_figures
from data.dataset import MockDataLoader, build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GST-VLA")
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--checkpoint",   default=None)
    parser.add_argument("--data_root",    default=None)
    parser.add_argument("--split",        default="val")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--n_batches",    type=int, default=None,
                        help="Limit to N batches (None=all)")
    parser.add_argument("--output_dir",   default="eval_results")
    parser.add_argument("--visualize",    action="store_true",
                        help="Generate paper figures")
    parser.add_argument("--n_viz_samples",type=int, default=4)
    parser.add_argument("--mock",         action="store_true",
                        help="Use mock model + data")
    parser.add_argument("--compare",      nargs="+", default=None,
                        metavar="NAME=CKPT",
                        help="Compare multiple checkpoints: name1=path1 name2=path2")
    parser.add_argument("--device",       default="auto")
    return parser.parse_args()


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_from_checkpoint(cfg, checkpoint_path, device, use_mock=False):
    m = cfg["model"]
    model = GSTVLA(
        d_sem=m["d_sem"],
        d_gst=m["d_gst"],
        N_g=m["N_g"],
        fourier_bands=m["fourier_bands"],
        img_size=m["img_size"],
        patch_size=m["patch_size"],
        d_vlm=m["d_vlm"],
        vlm_model_name=m["vlm_model"],
        d_state=m["d_state"],
        H=m["H"],
        d_action=m["d_action"],
        n_expert_layers=m["n_expert_layers"],
        use_mock_encoders=use_mock,
        use_mock_vlm=use_mock,
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        stage = ckpt.get("stage", "?")
        epoch = ckpt.get("epoch", "?")
        print(f"  [✓] Loaded checkpoint: {checkpoint_path}  (stage={stage}, epoch={epoch})")
    else:
        if checkpoint_path:
            print(f"  [!] Checkpoint not found: {checkpoint_path}. Using random weights.")

    model.eval()
    return model


def run_evaluation(
    model,
    dataloader,
    device,
    output_dir: str,
    model_name: str = "gst_vla",
    n_batches=None,
    visualize: bool = False,
    n_viz_samples: int = 4,
) -> EvalResults:
    """Run full evaluation for a single model."""

    print(f"\n{'─'*55}")
    print(f"  Evaluating: {model_name}")
    print(f"{'─'*55}")

    evaluator = GSTVLAEvaluator(model, device)
    results   = evaluator.evaluate(dataloader, n_batches=n_batches)

    # Print to console
    evaluator.print_report(results, title=f"GST-VLA Evaluation — {model_name}")

    # Save JSON
    out_dir = Path(output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluator.save_report(results, str(out_dir / "metrics.json"))

    # Visualize
    if visualize:
        print(f"\n  [Viz] Generating paper figures for {model_name}...")
        viz_dir = str(out_dir / "figures")
        generate_paper_figures(model, dataloader, device, viz_dir, n_viz_samples)

    return results


def compare_models(
    models_dict: dict,           # name → model
    dataloader,
    device,
    output_dir: str,
    n_batches=None,
) -> dict:
    """
    Evaluate multiple models and generate a comparison figure.
    """
    results_dict = {}

    for name, model in models_dict.items():
        results_dict[name] = run_evaluation(
            model, dataloader, device, output_dir,
            model_name=name, n_batches=n_batches,
        )

    # Comparison figure
    viz = GSTVisualizer(output_dir=output_dir, dark_mode=True)
    fig_path = viz.plot_eval_summary(
        results_dict,
        save_path=str(Path(output_dir) / "comparison_summary.png"),
    )
    if fig_path:
        print(f"\n  [✓] Comparison figure: {fig_path}")

    # Save comparison table
    comparison_table = {}
    for name, res in results_dict.items():
        comparison_table[name] = res.to_dict()

    table_path = str(Path(output_dir) / "comparison_table.json")
    with open(table_path, "w") as f:
        json.dump(comparison_table, f, indent=2)
    print(f"  [✓] Comparison table: {table_path}")

    # Print compact table
    print_comparison_table(results_dict)

    return results_dict


def print_comparison_table(results_dict: dict):
    """Print compact comparison table to console."""
    names   = list(results_dict.keys())
    metrics = [
        ("action_l2_mean",      "L2 ↓",       ".4f"),
        ("action_smoothness",   "Smooth ↓",    ".6f"),
        ("depth_absrel",        "AbsRel ↓",    ".4f"),
        ("depth_delta1",        "δ₁ ↑",        ".3f"),
        ("gst_coverage",        "Coverage ↑",  ".3f"),
        ("gst_entropy",         "Entropy ↑",   ".3f"),
        ("latency_mean_ms",     "Lat(ms) ↓",   ".1f"),
    ]

    col_w = 14
    header = f"{'Metric':<18}" + "".join(f"{n[:col_w]:>{col_w}}" for n in names)
    print("\n" + "─" * len(header))
    print("  COMPARISON TABLE")
    print("─" * len(header))
    print("  " + header)
    print("─" * len(header))

    for key, label, fmt in metrics:
        row = f"  {label:<18}"
        vals = [getattr(results_dict[n], key, 0.0) for n in names]
        for v in vals:
            row += f"{v:{fmt}:>{col_w}}"
        print(row)

    print("─" * len(header) + "\n")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto"
                          else args.device if args.device != "auto" else "cpu")
    print(f"\n[GST-VLA Eval] Device: {device} | Mock: {args.mock}")

    # Load config
    try:
        cfg = load_config(args.config)
    except Exception:
        # Fallback minimal config for mock mode
        cfg = {
            "model": {
                "d_sem": 1152, "d_gst": 512, "N_g": 128, "fourier_bands": 16,
                "img_size": 224, "patch_size": 14, "d_vlm": 3584,
                "vlm_model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "d_state": 14, "H": 16, "d_action": 7,
                "n_expert_layers": 8, "n_euler_steps": 10,
            }
        }

    # DataLoader
    if args.mock or args.data_root is None:
        print("[DataLoader] Using MOCK data")
        dataloader = MockDataLoader(
            batch_size=args.batch_size, num_batches=20,
            H=cfg["model"]["H"], d_state=cfg["model"]["d_state"],
        )
    else:
        dataloader = build_dataloader(
            args.data_root, args.split,
            batch_size=args.batch_size,
            num_workers=4,
            H=cfg["model"]["H"],
            d_state=cfg["model"]["d_state"],
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Compare multiple checkpoints ──────────────────
    if args.compare:
        models_dict = {}
        for spec in args.compare:
            name, ckpt_path = spec.split("=", 1)
            print(f"  Loading model [{name}]...")
            models_dict[name] = build_model_from_checkpoint(
                cfg, ckpt_path, device, use_mock=args.mock
            )

        compare_models(
            models_dict, dataloader, device, args.output_dir, args.n_batches
        )

    # ── Single model evaluation ────────────────────────
    else:
        model = build_model_from_checkpoint(
            cfg, args.checkpoint, device, use_mock=args.mock
        )
        run_evaluation(
            model, dataloader, device, args.output_dir,
            model_name="gst_vla",
            n_batches=args.n_batches,
            visualize=args.visualize,
            n_viz_samples=args.n_viz_samples,
        )

    print("\n[GST-VLA Eval] Done ✓")


if __name__ == "__main__":
    main()
