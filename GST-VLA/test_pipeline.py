"""
DEAD-VLA End-to-End Pipeline Feasibility Test
===============================================
ACCV 2026

Verifies correctness of the full pipeline in mock mode:
  - No large model weights required (SigLIP / DepthV2 / Qwen are mocked)
  - Runs on CPU (no GPU required)
  - Checks: tensor shapes, forward pass, backward pass, inference, parameter count

Usage:
    cd /home/omer/Projects/VLAs/GST-VLA-main/GST-VLA
    python test_pipeline.py
"""

import sys
import traceback
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent))

import torch
from models.gst_vla import GSTVLA
from training.losses import GSTVLALoss, GSTVLATrainer
from data.dataset import MockDataLoader


# ─────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────

def passed(msg: str):
    print(f"  ✓  {msg}")

def failed(msg: str, exc: Exception):
    print(f"  ✗  {msg}")
    traceback.print_exc()
    raise exc


# ─────────────────────────────────────────────
# 1. Parameter Summary
# ─────────────────────────────────────────────

def test_parameter_count():
    print("\n[1/7] Parameter Summary")
    print("─" * 50)
    try:
        model = GSTVLA(use_mock_encoders=True, use_mock_vlm=True)
        model.print_parameter_summary()

        stats = model.count_parameters()
        assert stats["gst"] > 0,            "GST has no trainable params"
        assert stats["vlm_projector"] > 0,  "VLM projector has no trainable params"
        assert stats["state_encoder"] > 0,  "State encoder has no trainable params"
        assert stats["da_cot"] > 0,         "DA-CoT has no trainable params"
        assert stats["action_expert"] > 0,  "Action expert has no trainable params"
        assert stats["total_trainable"] > 0, "No trainable params"
        passed("All components have trainable parameters")
        return model
    except Exception as e:
        failed("Parameter count test", e)


# ─────────────────────────────────────────────
# 2. Forward Pass (training)
# ─────────────────────────────────────────────

def test_forward_pass(model):
    print("\n[2/7] Forward Pass (training)")
    print("─" * 50)
    try:
        model.train()
        B = 2

        rgb         = torch.randn(B, 3, 224, 224)
        inst_ids    = torch.ones(B, 32, dtype=torch.long)
        attn_mask   = torch.ones(B, 32, dtype=torch.long)
        robot_state = torch.randn(B, 14)
        gt_actions  = torch.randn(B, 16, 7)

        out = model(rgb, inst_ids, attn_mask, robot_state, gt_actions)

        # Check required output keys
        for key in ["loss", "loss_flow", "loss_cot", "z_spatial", "h_vlm", "h_cot", "reasoning", "gst_aux"]:
            assert key in out, f"Missing output key: {key}"

        # Verify shapes
        assert out["z_spatial"].shape == (B, 128, 512),  f"z_spatial: {out['z_spatial'].shape}"
        assert out["h_vlm"].shape[0] == B,               f"h_vlm batch dim: {out['h_vlm'].shape}"
        assert out["h_cot"].shape[0] == B,               f"h_cot batch dim: {out['h_cot'].shape}"
        assert out["h_cot"].shape[1] > out["h_vlm"].shape[1], \
            f"h_cot should be larger than h_vlm (prepended CoT tokens)"

        # Reasoning output shapes
        r = out["reasoning"]
        assert r["obj_grounding"].shape    == (B, 8, 3), f"obj_grounding: {r['obj_grounding'].shape}"
        assert r["grasp_affordance"].shape == (B, 4, 3), f"grasp_affordance: {r['grasp_affordance'].shape}"
        assert r["spatial_relations"].shape== (B, 8, 6), f"spatial_relations: {r['spatial_relations'].shape}"
        assert r["motion_plan"].shape      == (B, 8, 7), f"motion_plan: {r['motion_plan'].shape}"

        # Loss should be a finite scalar
        loss = out["loss"]
        assert loss.dim() == 0,                   "Loss is not a scalar"
        assert torch.isfinite(loss),              f"Loss is not finite: {loss.item()}"

        print(f"     z_spatial  : {out['z_spatial'].shape}")
        print(f"     h_vlm      : {out['h_vlm'].shape}   (N_g={128} spatial + L=32 lang + 1 state)")
        print(f"     h_cot      : {out['h_cot'].shape}   (+ {model.da_cot.n_queries} CoT queries prepended)")
        print(f"     loss_flow  : {out['loss_flow'].item():.4f}")
        print(f"     loss_cot   : {out['loss_cot'].item():.4f}  (0 = no GT annotations)")
        print(f"     loss_total : {out['loss'].item():.4f}")
        passed("Forward pass shapes and values are correct")
        return out
    except Exception as e:
        failed("Forward pass test", e)


# ─────────────────────────────────────────────
# 3. Backward Pass (gradient flow)
# ─────────────────────────────────────────────

def test_backward_pass(model, out):
    print("\n[3/7] Backward Pass (gradient flow)")
    print("─" * 50)
    try:
        out["loss"].backward()

        # Verify gradients flow to key trainable modules
        grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name] = param.grad.norm().item()

        assert len(grads) > 0, "No gradients computed!"

        # Check key modules received gradients
        gst_grads    = {k: v for k, v in grads.items() if "gst." in k}
        cot_grads    = {k: v for k, v in grads.items() if "da_cot." in k}
        expert_grads = {k: v for k, v in grads.items() if "action_expert." in k}
        state_grads  = {k: v for k, v in grads.items() if "state_encoder." in k}

        assert len(gst_grads) > 0,    "No gradients in GST module"
        assert len(cot_grads) > 0,    "No gradients in DA-CoT module"
        assert len(expert_grads) > 0, "No gradients in Action Expert"
        assert len(state_grads) > 0,  "No gradients in State Encoder"

        print(f"     GST params with gradients:          {len(gst_grads)}")
        print(f"     DA-CoT params with gradients:       {len(cot_grads)}")
        print(f"     Action Expert params with gradients: {len(expert_grads)}")
        print(f"     State Encoder params with gradients: {len(state_grads)}")
        passed("Gradients flow to all trainable modules")

        # Reset gradients
        model.zero_grad()
    except Exception as e:
        failed("Backward pass test", e)


# ─────────────────────────────────────────────
# 4. Inference
# ─────────────────────────────────────────────

def test_inference(model):
    print("\n[4/7] Inference (predict_action)")
    print("─" * 50)
    try:
        model.eval()
        B = 1

        rgb         = torch.randn(B, 3, 224, 224)
        inst_ids    = torch.ones(B, 32, dtype=torch.long)
        attn_mask   = torch.ones(B, 32, dtype=torch.long)
        robot_state = torch.randn(B, 14)

        with torch.no_grad():
            actions, info = model.predict_action(rgb, inst_ids, attn_mask, robot_state)

        # Expected output shape: (B, H, d_action) = (1, 16, 7)
        assert actions.shape == (B, 16, 7), f"Expected (1,16,7), got {actions.shape}"
        assert torch.isfinite(actions).all(), "Actions contain non-finite values"

        assert "z_spatial"  in info
        assert "mu_3d"      in info
        assert "alpha"      in info
        assert "reasoning"  in info
        assert "h_cot"      in info

        print(f"     actions shape:  {actions.shape}  ✓  (B, H=16, d_action=7)")
        print(f"     mu_3d shape:    {info['mu_3d'].shape}")
        print(f"     alpha shape:    {info['alpha'].shape}")
        reasoning = info["reasoning"]
        print(f"     obj_grounding:  {reasoning['obj_grounding'].shape}")
        print(f"     motion_plan:    {reasoning['motion_plan'].shape}")
        passed("Inference produces correct output shapes")
        return actions
    except Exception as e:
        failed("Inference test", e)


# ─────────────────────────────────────────────
# 5. GST Fourier Dimension
# ─────────────────────────────────────────────

def test_fourier_dim():
    print("\n[5/7] GST Fourier Dimension Consistency")
    print("─" * 50)
    try:
        from models.gst import GaussianSpatialTokenizer
        from utils.fourier import FourierPositionalEncoding3D

        gst = GaussianSpatialTokenizer(fourier_bands=16)
        pe  = gst.pos_enc_3d

        pts = torch.randn(2, 256, 3)
        enc = pe(pts)

        expected_dim = 3 * 2 * 16  # 96
        assert enc.shape[-1] == expected_dim, \
            f"Fourier PE dim mismatch: got {enc.shape[-1]}, expected {expected_dim}"
        assert pe.include_raw == False, "include_raw should be False to match d_fourier=96"

        print(f"     fourier_bands=16  →  d_fourier={enc.shape[-1]}  (3×2×16, include_raw=False)")
        print(f"     d_in for attn_pool = {1152} + {enc.shape[-1]} = {1152 + enc.shape[-1]}")
        passed("Fourier dimension is consistent (no shape mismatch)")
    except Exception as e:
        failed("Fourier dimension test", e)


# ─────────────────────────────────────────────
# 6. Loss Function
# ─────────────────────────────────────────────

def test_loss_function():
    print("\n[6/7] Loss Function (GSTVLALoss)")
    print("─" * 50)
    try:
        criterion = GSTVLALoss(lambda_cot=0.1, lambda_depth=0.1)
        B = 2

        loss_flow = torch.tensor(0.5)
        loss_cot  = torch.tensor(0.3)
        gst_aux   = {
            "alpha":     torch.rand(B, 256, 1) * 0.5 + 0.1,   # (B, N, 1)
            "log_scale": torch.randn(B, 256, 3) * 0.1,         # (B, N, 3)
        }

        # Without CoT targets (unsupervised)
        losses = criterion(loss_flow=loss_flow, gst_aux=gst_aux)
        assert "loss_total" in losses
        assert "loss_opacity" in losses
        assert "loss_scale" in losses
        assert torch.isfinite(losses["loss_total"])
        print(f"     Without CoT targets: loss_total = {losses['loss_total'].item():.4f}")

        # With CoT loss
        losses2 = criterion(loss_flow=loss_flow, gst_aux=gst_aux, loss_cot=loss_cot)
        assert "loss_cot" in losses2
        assert losses2["loss_total"] > losses["loss_total"] - 1e-6  # CoT adds to total
        print(f"     With CoT loss:       loss_total = {losses2['loss_total'].item():.4f}")

        passed("Loss function computes correct combined loss")
    except Exception as e:
        failed("Loss function test", e)


# ─────────────────────────────────────────────
# 7. Mock DataLoader + Trainer Integration
# ─────────────────────────────────────────────

def test_trainer_integration():
    print("\n[7/7] Trainer Integration (1 training step)")
    print("─" * 50)
    try:
        device = torch.device("cpu")
        model  = GSTVLA(use_mock_encoders=True, use_mock_vlm=True).to(device)

        cfg = {
            "lambda_cot": 0.1, "lambda_depth": 0.1,
            "lambda_opacity": 0.01, "lambda_scale": 0.001,
        }
        trainer = GSTVLATrainer(model, cfg, device)
        trainer.set_stage(1)

        loader = MockDataLoader(batch_size=2, num_batches=2, H=16, d_state=14)
        batch  = next(iter(loader))

        # Use CPU autocast context (no-op on CPU for bf16, but should not error)
        with torch.amp.autocast("cpu", enabled=False):
            model.train()
            trainer.optimizer.zero_grad()
            out = model(
                rgb=batch["rgb"],
                instruction_ids=batch["instruction_ids"],
                attention_mask=batch["attention_mask"],
                robot_state=batch["robot_state"],
                gt_actions=batch["actions"],
            )
            losses = trainer.criterion(
                loss_flow=out["loss_flow"],
                gst_aux=out["gst_aux"],
                loss_cot=out.get("loss_cot"),
            )
            losses["loss_total"].backward()
            trainer.optimizer.step()

        assert torch.isfinite(losses["loss_total"]), "Training loss is not finite"
        print(f"     train step loss_total = {losses['loss_total'].item():.4f}")
        passed("Trainer integration step succeeded")
    except Exception as e:
        failed("Trainer integration test", e)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DEAD-VLA Pipeline Feasibility Test")
    print("  ACCV 2026 — Mock Mode (CPU, no model weights)")
    print("=" * 60)

    n_passed = 0
    n_total  = 7

    try:
        model = test_parameter_count()
        n_passed += 1

        test_fourier_dim()
        n_passed += 1

        out = test_forward_pass(model)
        n_passed += 1

        test_backward_pass(model, out)
        n_passed += 1

        test_inference(model)
        n_passed += 1

        test_loss_function()
        n_passed += 1

        test_trainer_integration()
        n_passed += 1

    except Exception:
        pass

    print("\n" + "=" * 60)
    print(f"  RESULTS: {n_passed}/{n_total} tests passed")
    if n_passed == n_total:
        print("  ALL TESTS PASSED ✓")
        print("\n  Pipeline is FEASIBLE and ARCHITECTURALLY CORRECT.")
        print("  Next steps:")
        print("    1. Download model weights (SigLIP, DepthV2, Qwen2.5-VL-7B)")
        print("    2. Prepare dataset via data_loaders/ converters")
        print("    3. Run: python train.py --stage 1 --data_root /path/to/data")
    else:
        print(f"  {n_total - n_passed} test(s) FAILED. See errors above.")
    print("=" * 60)
    return n_passed == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
