# GST-VLA: Structured Gaussian Spatial Tokens for 3D-Aware Vision-Language-Action Models
## ACCV 2026 Submission

### Architecture Overview
```
RGB Image  ──► Semantic Encoder (SigLIP ViT-SO400M/14) [FROZEN]
               ├── patch tokens: R^(256×1152)
Depth Map  ──► Depth Expert (Depth Anything V2 ViT-L)  [FROZEN]
               └── metric depth: R^(H×W)
Language   ──► BPE Tokenizer                           [FROZEN]
Robot State ──►

All ──► Gaussian Spatial Tokenizer (GST) [TRAINABLE - NOVEL]
        ① Back-project to 3D:  p_i = D_i * K^{-1} [u,v,1]^T
        ② Gaussian Parameters: μ, Σ, α = MLP(f_sem)
        ③ 3D Positional Encoding: γ(p) = [sin,cos]^L Fourier
        ④ Spatial Aggregation: Radius grouping + attn pool
        → z_spatial ∈ R^(N_g × d)

z_spatial ──► VLM Reasoning Core (Qwen2.5-VL 7B) [FROZEN for ACCV]
              Cross-attention projector W_proj: d_gst → d_vlm
              → h_vlm ∈ R^(N×d_vlm)

h_vlm ──► Flow Matching Action Expert [TRAINABLE]
           300M params, N=8 layers, d=512
           Separate expert (MoE-style)
           FiLM conditioning: h_vlm → adaLN, s_t → FiLM
           → a_t ∈ R^(16×7)  [Δpose(6)+gripper(1)×H]
```

### Project Structure
```
gst_vla/
├── models/
│   ├── gst.py              # Gaussian Spatial Tokenizer (CORE NOVEL MODULE)
│   ├── encoders.py         # SigLIP + Depth Anything V2 wrappers
│   ├── vlm_wrapper.py      # Frozen Qwen2.5-VL interface
│   ├── flow_matching.py    # Flow Matching Action Expert
│   └── gst_vla.py          # Full model assembly
├── training/
│   ├── trainer.py          # 3-stage training pipeline
│   └── losses.py           # Combined loss functions
├── data/
│   └── dataset.py          # Robot manipulation dataset
├── utils/
│   ├── geometry.py         # 3D projection utilities
│   └── fourier.py          # Positional encodings
├── configs/
│   └── default.yaml        # Hyperparameters
└── README.md
```

### Training Stages (3-Stage)
- **S1**: GST + Depth Expert (frozen VLM + frozen SigLIP)
- **S2**: LoRA on VLM projector + Flow Expert
- **S3**: Full fine-tune (GST + projector + expert)

### Key Contributions (ACCV)
- **C1**: Gaussian Spatial Tokenizer — explicit 3D tokens from depth+RGB
- **C2**: 3D-conditioned flow matching for robot action prediction
- **C3**: 3-stage training pipeline

### Incremental (ICRA - Next)
- **C4**: Depth-Aware Chain-of-Thought (DA-CoT) reasoning block
