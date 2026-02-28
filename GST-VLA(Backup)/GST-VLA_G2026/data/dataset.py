import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Dict, Any

class BridgeV2Dataset(Dataset):
    def __init__(
        self, 
        data_directory: str, 
        split: str = "train",
        chunk_size: int = 16,     # H=16 from your diagram
        action_dim: int = 7,      # Delta XYZ (3) + Euler RPY (3) + Gripper (1)
        image_size: int = 224     # Required by SigLIP/DepthAnything
    ):
        """
        PyTorch Dataset for Bridge Data V2 (assumes pre-processed offline format like Zarr/HDF5 or a list of episode dicts).
        Handles sequence batching and Action Chunking for Flow Matching.
        """
        super().__init__()
        self.data_directory = data_directory
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        
        # Load your dataset index here (e.g., list of file paths or Zarr group keys)
        # For this skeleton, we assume self.episodes is a list of episode metadata
        self.episodes = self._load_episode_index(data_directory, split)
        
        # Create a unified index mapping global step `i` to (episode_id, step_t)
        self.step_index = self._build_step_index()

        # Image transforms for SigLIP & DepthAnything V2
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            # Standard SigLIP normalization
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

    def _load_episode_index(self, path: str, split: str) -> list:
        """Mock function: Load episode metadata."""
        # Returns a list of dicts: [{'path': 'epi_0.zarr', 'length': 45}, ...]
        return [{'id': i, 'length': 50} for i in range(100)] # Mock data

    def _build_step_index(self) -> list:
        """Flattens episodes into a list of (episode_idx, frame_idx)."""
        index = []
        for epi_idx, epi in enumerate(self.episodes):
            # We can sample up to the very last frame. Action chunking will pad if necessary.
            for t in range(epi['length']):
                index.append((epi_idx, t))
        return index

    def __len__(self) -> int:
        return len(self.step_index)

    def _get_episode_data(self, episode_idx: int) -> Dict[str, np.ndarray]:
        """
        Mock function: Load the actual episode tensors from disk.
        Bridge V2 standard data structure:
        - 'image': (T, H, W, 3)
        - 'language': string
        - 'state': (T, 14) -> e.g., 7D joint positions + 7D end-effector pose
        - 'action': (T, 7) -> Delta XYZ, Delta RPY, Gripper state (0/1)
        - 'intrinsics': (3, 3) camera matrix
        """
        length = self.episodes[episode_idx]['length']
        return {
            "image": np.random.randint(0, 255, (length, 256, 256, 3), dtype=np.uint8),
            "language": "put the sweet potato on the plate", # Classic Bridge V2 task
            "state": np.random.randn(length, 14).astype(np.float32),
            "action": np.random.randn(length, self.action_dim).astype(np.float32),
            "intrinsics": np.eye(3, dtype=np.float32) * 500 # Mock focal length
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        epi_idx, t = self.step_index[idx]
        
        # 1. Load Episode
        epi_data = self._get_episode_data(epi_idx)
        epi_len = len(epi_data["image"])
        
        # 2. Extract Current Step Modalities (t)
        # Image
        img_np = epi_data["image"][t]
        img_pil = Image.fromarray(img_np)
        rgb_image = self.image_transform(img_pil) # (3, 224, 224)
        
        # Language & State
        text_prompt = epi_data["language"]
        robot_state = torch.from_numpy(epi_data["state"][t]) # (14,)
        intrinsics = torch.from_numpy(epi_data["intrinsics"]) # (3, 3)
        
        # 3. ACTION CHUNKING (t to t + chunk_size)
        # We need to extract H=16 actions starting from current time step t.
        # If the episode ends before t+16, we repeat the final state/action (padding).
        end_idx = t + self.chunk_size
        
        if end_idx <= epi_len:
            # We have enough steps remaining
            action_chunk = epi_data["action"][t : end_idx]
        else:
            # Pad the sequence by repeating the final action (usually a resting/holding state)
            valid_actions = epi_data["action"][t : epi_len]
            padding_len = end_idx - epi_len
            final_action = valid_actions[-1:] # (1, 7)
            padded_actions = np.repeat(final_action, padding_len, axis=0) # (padding_len, 7)
            action_chunk = np.concatenate([valid_actions, padded_actions], axis=0) # (16, 7)
            
        action_chunk_tensor = torch.from_numpy(action_chunk) # (16, 7)
        
        return {
            "rgb_image": rgb_image,             # (3, 224, 224)
            "intrinsics": intrinsics,           # (3, 3)
            "text_prompt": text_prompt,         # str
            "robot_state": robot_state,         # (14,)
            "action_chunk": action_chunk_tensor # (16, 7)
        }

# --- Quick PyTorch DataLoader Test ---
if __name__ == "__main__":
    dataset = BridgeV2Dataset(data_directory="./mock_bridge_data")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    batch = next(iter(dataloader))
    print("Batch loaded successfully!")
    print(f"Image shape: {batch['rgb_image'].shape}")          # [4, 3, 224, 224]
    print(f"Action Chunk shape: {batch['action_chunk'].shape}") # [4, 16, 7]
    print(f"Text Prompts: {batch['text_prompt']}")