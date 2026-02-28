# GST-VLA
GST-VLA: Structured Gaussian Spatial Tokens for 3D-Aware Vision-Language-Action Models
<img width="5040" height="2504" alt="DEAD-VLA" src="https://github.com/user-attachments/assets/a2b5de3c-2c6b-4888-be51-895f2f766423" />


  Run commands (all from GST-VLA/):

  # 1. Feasibility test — no model weights, CPU only
  python test_pipeline.py

  # 2. Mock training (smoke test all 3 stages)
  python train.py --mock --run_all_stages

  # 3. Real Stage 1 — once SigLIP + Qwen are available
  python train.py --stage 1 --data_root /path/to/data --batch_size 4

  # 4. Stage 2 — after stage 1 checkpoint
  python train.py --stage 2 --data_root /path/to/data \
    --resume checkpoints/stage1/best.pt

  # 5. Stage 3
  python train.py --stage 3 --data_root /path/to/data \
    --resume checkpoints/stage2/best.pt