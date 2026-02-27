"""
GST-VLA Main Training Script
==============================
ACCV 2026 Submission

Usage:
    # Quick test (mock mode)
    python train.py --mock --stage 1 --epochs 2

    # Stage 1 training
    python train.py --data_root /path/to/data --stage 1 --epochs 10

    # Stage 2 (after stage 1 checkpoint)
    python train.py --data_root /path/to/data --stage 2 --resume ckpt/stage1_best.pt

    # Full 3-stage pipeline
    python train.py --data_root /path/to/data --run_all_stages
"""

import os
import sys
import argparse
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.gst_vla import GSTVLA
from training.losses import GSTVLATrainer
from data.dataset import MockDataLoader, build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train GST-VLA")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--data_root",  default=None)
    parser.add_argument("--stage",      type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resume",     default=None, help="checkpoint path")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--mock",       action="store_true", help="Use mock encoders/VLM for testing")
    parser.add_argument("--run_all_stages", action="store_true")
    parser.add_argument("--device",     default="auto")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, use_mock: bool = False) -> GSTVLA:
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
        n_euler_steps=m["n_euler_steps"],
        use_mock_encoders=use_mock,
        use_mock_vlm=use_mock,
    )
    return model


def save_checkpoint(model, optimizer, epoch, stage, path: str, metrics: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "stage":          stage,
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "metrics":        metrics or {},
    }
    torch.save(checkpoint, path)
    print(f"  [✓] Checkpoint saved: {path}")


def load_checkpoint(model, path: str, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    if optimizer and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  [✓] Loaded checkpoint from: {path} (stage={ckpt.get('stage')}, epoch={ckpt.get('epoch')})")
    return ckpt


def run_stage(
    model: GSTVLA,
    trainer: GSTVLATrainer,
    stage: int,
    train_loader,
    val_loader,
    epochs: int,
    output_dir: str,
):
    """Run one training stage."""
    trainer.set_stage(stage)

    best_val_loss = float("inf")
    stage_dir = Path(output_dir) / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────
        model.train()
        train_losses = {"loss_total": 0.0, "loss_flow": 0.0}
        n_batches = 0

        for batch in train_loader:
            step_losses = trainer.train_step(batch)
            for k, v in step_losses.items():
                train_losses[k] = train_losses.get(k, 0.0) + v
            n_batches += 1

        # Average
        train_avg = {f"train/{k}": v / n_batches for k, v in train_losses.items()}

        # ── Validate ──────────────────────────────────
        model.eval()
        val_losses = {"loss_total": 0.0}
        n_val = 0

        for batch in val_loader:
            step_losses = trainer.eval_step(batch)
            for k, v in step_losses.items():
                val_losses[k] = val_losses.get(k, 0.0) + v
            n_val += 1

        val_avg = {f"val/{k}": v / max(n_val, 1) for k, v in val_losses.items()}
        val_loss = val_avg.get("val/loss_total", float("inf"))

        # ── Log ───────────────────────────────────────
        all_metrics = {**train_avg, **val_avg}
        print(
            f"  Stage {stage} | Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_avg.get('train/loss_total', 0):.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        # ── Save ──────────────────────────────────────
        save_checkpoint(
            model,
            trainer.optimizer,
            epoch,
            stage,
            str(stage_dir / f"epoch_{epoch:03d}.pt"),
            all_metrics,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                trainer.optimizer,
                epoch,
                stage,
                str(stage_dir / "best.pt"),
                all_metrics,
            )
            print(f"  [★] New best: {best_val_loss:.4f}")

    return best_val_loss


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n[GST-VLA] Device: {device}")
    print(f"[GST-VLA] Mock mode: {args.mock}\n")

    # Build model
    model = build_model(cfg, use_mock=args.mock)
    model = model.to(device)
    model.print_parameter_summary()

    # Load checkpoint if resuming
    if args.resume:
        load_checkpoint(model, args.resume)

    # Trainer
    trainer = GSTVLATrainer(model, cfg.get("loss", {}), device)

    # DataLoaders
    if args.mock or args.data_root is None:
        print("[DataLoader] Using MOCK data for testing")
        train_loader = MockDataLoader(
            batch_size=args.batch_size, num_batches=20,
            H=cfg["model"]["H"], d_state=cfg["model"]["d_state"],
        )
        val_loader = MockDataLoader(
            batch_size=args.batch_size, num_batches=5,
            H=cfg["model"]["H"], d_state=cfg["model"]["d_state"],
        )
    else:
        train_loader = build_dataloader(
            args.data_root, "train",
            batch_size=args.batch_size,
            num_workers=cfg["data"]["num_workers"],
            H=cfg["model"]["H"],
            d_state=cfg["model"]["d_state"],
        )
        val_loader = build_dataloader(
            args.data_root, "val",
            batch_size=args.batch_size,
            num_workers=cfg["data"]["num_workers"],
            H=cfg["model"]["H"],
            d_state=cfg["model"]["d_state"],
        )

    # Run training
    stage_epochs = cfg["training"]["stages"]

    if args.run_all_stages:
        for stage in [1, 2, 3]:
            epochs = args.epochs or stage_epochs[stage]["epochs"]
            print(f"\n{'='*60}")
            print(f"  STAGE {stage}: {stage_epochs[stage]['description']}")
            print(f"{'='*60}")
            run_stage(model, trainer, stage, train_loader, val_loader, epochs, args.output_dir)
    else:
        stage  = args.stage
        epochs = args.epochs or stage_epochs[stage]["epochs"]
        print(f"\n{'='*60}")
        print(f"  STAGE {stage}: {stage_epochs[stage]['description']}")
        print(f"{'='*60}")
        run_stage(model, trainer, stage, train_loader, val_loader, epochs, args.output_dir)

    print("\n[GST-VLA] Training complete! ✓")


if __name__ == "__main__":
    main()
