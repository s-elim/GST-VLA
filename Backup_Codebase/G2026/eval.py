import torch
import numpy as np
import time
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Import the model
from models.vla_model import GST_VLA



class MockRoboticEnv:
    """
    A generic wrapper representing simulation environments like RLBench, CALVIN, or real robot APIs (e.g., ROS2/Polymetis).
    """
    def __init__(self, task_name="pick_and_place", max_steps=200):
        self.task_name = task_name
        self.max_steps = max_steps
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        # Returns: RGB Image, Depth (optional for real envs if using DepthAnything), Intrinsics, Proprioception
        return self._get_obs()

    def step(self, action: np.ndarray):
        """Executes a single 7-DoF action in the environment."""
        self.current_step += 1
        
        # Mock success condition: succeeds 85% of the time if it reaches max_steps
        done = self.current_step >= self.max_steps
        success = True if (done and np.random.rand() > 0.15) else False
        
        return self._get_obs(), success, done

    def _get_obs(self):
        # Returns mock observations matching our dataset format
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        intrinsics = np.eye(3, dtype=np.float32) * 500
        state = np.random.randn(14).astype(np.float32)
        return rgb, intrinsics, state

def evaluate_gst_vla(checkpoint_path: str):
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = 16
    execution_horizon = 8  # Execute first 8 steps of the 16-step chunk, then replan
    euler_steps = 10       # From your Flow Matching specification
    num_episodes = 100
    
    # Standard image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # ==========================================
    # 2. Load Model & Checkpoint
    # ==========================================
    print("Loading GST-VLA Model for Evaluation...")
    model = GST_VLA(action_dim=7, chunk_size=chunk_size, device=device).to(device)
    
    # Load trained weights
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Ensure dropout and batch norm are frozen
    
    # ==========================================
    # 3. Initialize Environment
    # ==========================================
    env = MockRoboticEnv(task_name="stack_blocks")
    language_instruction = "stack the red block on the blue block"
    
    success_count = 0
    latency_logs = []

    print(f"\nStarting Evaluation over {num_episodes} episodes...")
    
    # ==========================================
    # 4. Evaluation Loop
    # ==========================================
    for ep in tqdm(range(num_episodes), desc="Evaluating Episodes"):
        rgb_np, intrinsics_np, state_np = env.reset()
        done = False
        success = False
        
        while not done:
            # Prepare inputs
            rgb_tensor = transform(rgb_np).unsqueeze(0).to(device) # (1, 3, 224, 224)
            intrinsics_tensor = torch.from_numpy(intrinsics_np).unsqueeze(0).to(device) # (1, 3, 3)
            state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(device) # (1, 14)
            
            # --- MEASURE INFERENCE LATENCY ---
            start_time = time.perf_counter()
            
            with torch.no_grad():
                # We use torch.amp.autocast to keep the 7B VLM fast during inference
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    # forward() without actions_target triggers generate_actions() (Euler Integration)
                    # Output shape: (1, 16, 7)
                    action_chunk = model(
                        rgb_image=rgb_tensor,
                        intrinsics=intrinsics_tensor,
                        text_prompts=[language_instruction],
                        robot_state=state_tensor
                    )
            
            # Ensure CUDA operations are finished before stopping the clock
            torch.cuda.synchronize() 
            end_time = time.perf_counter()
            latency_logs.append(end_time - start_time)
            
            # Extract the numpy array
            action_chunk_np = action_chunk.squeeze(0).cpu().numpy() # (16, 7)
            
            # --- RECEDING HORIZON EXECUTION ---
            # Execute only the first `execution_horizon` steps
            for i in range(execution_horizon):
                action_step = action_chunk_np[i]
                rgb_np, intrinsics_np, state_np, success, done = _env_step_wrapper(env, action_step)
                if done:
                    break
        
        if success:
            success_count += 1

    # ==========================================
    # 5. Report Metrics
    # ==========================================
    success_rate = (success_count / num_episodes) * 100
    avg_latency = np.mean(latency_logs) * 1000 # Convert to ms
    hz = 1000.0 / avg_latency
    
    print("\n" + "="*40)
    print("🏆 ACCV EVALUATION RESULTS 🏆")
    print("="*40)
    print(f"Task: {env.task_name}")
    print(f"Success Rate:      {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"Avg Inference Latency: {avg_latency:.1f} ms")
    print(f"Control Frequency:     {hz:.1f} Hz")
    print("="*40)
    
    # Validating your ~200ms target claim
    if avg_latency <= 250:
        print("✅ Latency target met! Ready for real-time control.")
    else:
        print("⚠️ Latency higher than expected. Consider reducing Euler steps or GST tokens.")

def _env_step_wrapper(env, action):
    """Helper to unpack env returns for the loop."""
    obs, success, done = env.step(action)
    return obs[0], obs[1], obs[2], success, done

if __name__ == "__main__":
    evaluate_gst_vla(checkpoint_path="./checkpoints/gst_vla_epoch_50.pt")