"""
DEAD-VLA Main Training Entry Point
=====================================
ACCV 2026

Usage:
    # Quick smoke test (mock encoders + VLM, no data required)
    python train.py --mock --stage 1 --epochs 2

    # Stage 1 training
    python train.py --data_root /path/to/data --stage 1 --epochs 10

    # Stage 2 (continue from stage 1 checkpoint)
    python train.py --data_root /path/to/data --stage 2 --resume checkpoints/stage1/best.pt

    # Run all 3 stages sequentially
    python train.py --data_root /path/to/data --run_all_stages
"""

import os
import sys
import argparse
import torch
import yaml
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent))

from models.gst_vla import GSTVLA
from training.losses import GSTVLATrainer
from data.dataset import MockDataLoader, build_dataloader


def parse_args():
    p = argparse.ArgumentParser(description="Train DEAD-VLA")
    p.add_argument("--config",          default="configs/default.yaml")
    p.add_argument("--data_root",       default=None)
    p.add_argument("--stage",           type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--epochs",          type=int, default=None,
                   help="Override config epoch count for the given stage")
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--resume",          default=None, help="Checkpoint path to resume from")
    p.add_argument("--output_dir",      default="./checkpoints")
    p.add_argument("--mock",            action="store_true",
                   help="Use mock encoders & VLM (for testing without GPU/model weights)")
    p.add_argument("--run_all_stages",  action="store_true",
                   help="Run all 3 training stages sequentially")
    p.add_argument("--device",          default="auto")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, use_mock: bool = False) -> GSTVLA:
    m = cfg["model"]
    return GSTVLA(
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
        n_cot_queries=m.get("n_cot_queries", 16),
        d_cot=m.get("d_cot", 512),
        K_obj=m.get("K_obj", 8),
        K_grasp=m.get("K_grasp", 4),
        K_rel=m.get("K_rel", 8),
        K_wp=m.get("K_wp", 8),
        use_mock_encoders=use_mock,
        use_mock_vlm=use_mock,
        depth_pretrained_path=m.get("depth_pretrained_path"),
    )


def save_checkpoint(model, optimizer, epoch, stage, path: str, metrics: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "stage":           stage,
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "metrics":         metrics or {},
    }, path)
    print(f"  [✓] Checkpoint saved: {path}")


def load_checkpoint(model, path: str, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    if optimizer and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  [✓] Loaded checkpoint: {path}  (stage={ckpt.get('stage')}, epoch={ckpt.get('epoch')})")
    return ckpt


def run_stage(model, trainer, stage, train_loader, val_loader, epochs, output_dir):
    """Run a single training stage."""
    trainer.set_stage(stage)
    best_val = float("inf")
    stage_dir = Path(output_dir) / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_acc, n_train = {}, 0
        for batch in train_loader:
            step = trainer.train_step(batch)
            for k, v in step.items():
                train_acc[k] = train_acc.get(k, 0.0) + v
            n_train += 1

        # Validate
        model.eval()
        val_acc, n_val = {}, 0
        for batch in val_loader:
            step = trainer.eval_step(batch)
            for k, v in step.items():
                val_acc[k] = val_acc.get(k, 0.0) + v
            n_val += 1

        val_total = val_acc.get("loss_total", 0.0) / max(n_val, 1)
        train_total = train_acc.get("loss_total", 0.0) / max(n_train, 1)
        print(
            f"  Stage {stage} | Epoch {epoch:3d}/{epochs} | "
            f"train={train_total:.4f} | val={val_total:.4f}"
        )

        metrics = {
            **{f"train/{k}": v / n_train for k, v in train_acc.items()},
            **{f"val/{k}": v / max(n_val, 1) for k, v in val_acc.items()},
        }
        save_checkpoint(model, trainer.optimizer, epoch, stage,
                        str(stage_dir / f"epoch_{epoch:03d}.pt"), metrics)

        if val_total < best_val:
            best_val = val_total
            save_checkpoint(model, trainer.optimizer, epoch, stage,
                            str(stage_dir / "best.pt"), metrics)
            print(f"  [★] New best val loss: {best_val:.4f}")

    return best_val


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    print(f"\n[DEAD-VLA] Device:    {device}")
    print(f"[DEAD-VLA] Mock mode: {args.mock}\n")

    model = build_model(cfg, use_mock=args.mock).to(device)
    model.print_parameter_summary()

    if args.resume:
        load_checkpoint(model, args.resume)

    trainer = GSTVLATrainer(model, cfg.get("loss", {}), device)

    if args.mock or args.data_root is None:
        print("[DataLoader] Using MOCK data")
        H, d = cfg["model"]["H"], cfg["model"]["d_state"]
        train_loader = MockDataLoader(batch_size=args.batch_size, num_batches=20, H=H, d_state=d)
        val_loader   = MockDataLoader(batch_size=args.batch_size, num_batches=5,  H=H, d_state=d)
    else:
        kw = dict(H=cfg["model"]["H"], d_state=cfg["model"]["d_state"],
                  num_workers=cfg["data"]["num_workers"])
        train_loader = build_dataloader(args.data_root, "train", args.batch_size, **kw)
        val_loader   = build_dataloader(args.data_root, "val",   args.batch_size, **kw)

    stage_cfgs = cfg["training"]["stages"]

    if args.run_all_stages:
        for s in [1, 2, 3]:
            epochs = args.epochs or stage_cfgs[s]["epochs"]
            print(f"\n{'='*60}\n  STAGE {s}: {stage_cfgs[s]['description']}\n{'='*60}")
            run_stage(model, trainer, s, train_loader, val_loader, epochs, args.output_dir)
    else:
        s = args.stage
        epochs = args.epochs or stage_cfgs[s]["epochs"]
        print(f"\n{'='*60}\n  STAGE {s}: {stage_cfgs[s]['description']}\n{'='*60}")
        run_stage(model, trainer, s, train_loader, val_loader, epochs, args.output_dir)

    print("\n[DEAD-VLA] Training complete ✓")


if __name__ == "__main__":
    main()
