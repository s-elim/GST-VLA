import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb # For logging

# Import modules from our previous parts
from models.vla_model import GST_VLA
from data.dataset import BridgeV2Dataset

def train_gst_vla():
    # ==========================================
    # 1. Configuration & Hyperparameters
    # ==========================================
    config = {
        "epochs": 50,
        "batch_size": 16,            # Adjust based on A100 VRAM
        "lr": 1e-4,                  # Standard for DiT / Flow Matching
        "weight_decay": 1e-3,
        "chunk_size": 16,            # H=16
        "action_dim": 7,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "./checkpoints"
    }
    
    os.makedirs(config["save_dir"], exist_ok=True)
    wandb.init(project="GST-VLA-ACCV", config=config)

    # ==========================================
    # 2. Initialize Dataset & DataLoader
    # ==========================================
    print("Loading Bridge V2 Dataset...")
    train_dataset = BridgeV2Dataset(
        data_directory="./bridge_v2_data", # Path to your processed data
        split="train",
        chunk_size=config["chunk_size"],
        action_dim=config["action_dim"]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )

    # ==========================================
    # 3. Initialize Model
    # ==========================================
    print("Initializing GST-VLA Model (Freezing backbones)...")
    model = GST_VLA(
        action_dim=config["action_dim"],
        chunk_size=config["chunk_size"],
        device=torch.device(config["device"])
    ).to(config["device"])

    # ==========================================
    # 4. Optimizer setup (CRITICAL FOR EFFICIENCY)
    # ==========================================
    # We ONLY pass parameters that require gradients to the optimizer.
    # This prevents the optimizer from allocating momentum states for the 7B VLM.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {num_total / 1e9:.2f}B")
    print(f"Trainable Parameters: {num_trainable / 1e6:.2f}M") # Should be ~300M + GST params

    optimizer = AdamW(trainable_params, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"] * len(train_loader))
    
    # Mixed precision scaler for memory efficiency
    scaler = torch.amp.GradScaler('cuda')

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print("Starting Flow Matching Training...")
    model.train()
    
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to GPU
            rgb_image = batch["rgb_image"].to(config["device"])
            intrinsics = batch["intrinsics"].to(config["device"])
            text_prompts = batch["text_prompt"] # List of strings, handled by tokenizer inside wrapper
            robot_state = batch["robot_state"].to(config["device"])
            actions_target = batch["action_chunk"].to(config["device"])
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with Automatic Mixed Precision (AMP)
            # This is crucial for evaluating the 7B model without OOM errors
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # When actions_target is passed, the model computes the Flow Matching MSE loss
                loss = model(
                    rgb_image=rgb_image,
                    intrinsics=intrinsics,
                    text_prompts=text_prompts,
                    robot_state=robot_state,
                    actions_target=actions_target
                )
            
            # Backward pass and optimization
            scaler.scale(loss).backward()
            
            # Gradient clipping to stabilize DiT training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Logging
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})
            
            if batch_idx % 50 == 0:
                wandb.log({
                    "train_loss": loss.item(), 
                    "learning_rate": scheduler.get_last_lr()[0],
                    "step": epoch * len(train_loader) + batch_idx
                })

        # Save checkpoint at the end of each epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Completed. Average Loss: {avg_loss:.4f}")
        
        ckpt_path = os.path.join(config["save_dir"], f"gst_vla_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)
        
    wandb.finish()
    print("Training Complete!")

if __name__ == "__main__":
    train_gst_vla()