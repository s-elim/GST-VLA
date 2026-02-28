"""
Dataset for GST-VLA Training
==============================
Supports LIBERO, MetaWorld, RLBench formatted datasets.

Each sample provides:
    rgb:            (3, 224, 224) normalized RGB image
    depth_gt:       (224, 224)   metric depth (if available)
    instruction_ids:(L,)         tokenized language instruction
    attention_mask: (L,)         language attention mask
    robot_state:    (d_state,)   joint positions + velocities
    actions:        (H, 7)       action chunk [Δpose(6), gripper(1)]
    camera_K:       (3, 3)       camera intrinsics (optional)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import json


class RobotManipulationDataset(Dataset):
    """
    Generic robot manipulation dataset for GST-VLA.
    
    Expects data organized as:
        root/
          episode_000/
            rgb/         frame_000.png, frame_001.png, ...
            depth/       frame_000.npy, ...  (optional)
            actions.npy  (T, 7) action sequence
            instruction.txt
            robot_states.npy  (T, d_state)
            camera_K.npy      (3, 3)  optional
          episode_001/
          ...
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        H: int = 16,               # action chunk horizon
        d_state: int = 14,
        img_size: int = 224,
        max_instruction_len: int = 64,
        tokenizer=None,            # Optional tokenizer (or mock)
        transform=None,
        use_mock_depth: bool = False,
        depth_max: float = 5.0,
    ):
        self.data_root = Path(data_root)
        self.split     = split
        self.H         = H
        self.d_state   = d_state
        self.img_size  = img_size
        self.max_len   = max_instruction_len
        self.tokenizer = tokenizer
        self.transform = transform
        self.use_mock_depth = use_mock_depth
        self.depth_max = depth_max

        # Load episode index
        self.episodes = self._load_episode_index()

        # Build (episode_idx, frame_idx) pairs
        self.samples = self._build_sample_index()

    def _load_episode_index(self) -> List[Path]:
        """Find all episode directories."""
        split_file = self.data_root / f"{self.split}_episodes.json"
        if split_file.exists():
            with open(split_file) as f:
                episode_names = json.load(f)
            return [self.data_root / ep for ep in episode_names]
        else:
            # Fallback: all subdirs
            episodes = sorted([
                d for d in self.data_root.iterdir()
                if d.is_dir() and d.name.startswith("episode")
            ])
            # Simple train/val split
            n = len(episodes)
            if self.split == "train":
                return episodes[:int(0.9 * n)]
            else:
                return episodes[int(0.9 * n):]

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """Build (episode_idx, timestep_idx) index."""
        samples = []
        for ep_idx, ep_dir in enumerate(self.episodes):
            actions_path = ep_dir / "actions.npy"
            if not actions_path.exists():
                continue
            actions = np.load(actions_path)  # (T, 7)
            T = len(actions)
            for t in range(T - self.H):
                samples.append((ep_idx, t))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self.samples[idx]
        ep_dir = self.episodes[ep_idx]

        # ── RGB ──────────────────────────────────────────
        rgb = self._load_rgb(ep_dir, t)  # (3, 224, 224)

        # ── Depth ────────────────────────────────────────
        depth = self._load_depth(ep_dir, t)  # (224, 224) or mock

        # ── Robot State ───────────────────────────────────
        robot_state = self._load_robot_state(ep_dir, t)  # (d_state,)

        # ── Actions (chunk) ───────────────────────────────
        actions = self._load_action_chunk(ep_dir, t)  # (H, 7)

        # ── Instruction ────────────────────────────────────
        instruction_ids, attention_mask = self._load_instruction(ep_dir)

        # ── Camera Intrinsics (optional) ──────────────────
        K = self._load_camera_K(ep_dir)  # (3, 3) or None

        sample = {
            "rgb":              rgb,
            "depth_gt":         depth,
            "instruction_ids":  instruction_ids,
            "attention_mask":   attention_mask,
            "robot_state":      robot_state,
            "actions":          actions,
        }
        if K is not None:
            sample["camera_K"] = K

        return sample

    def _load_rgb(self, ep_dir: Path, t: int) -> torch.Tensor:
        rgb_dir = ep_dir / "rgb"
        frame_path = rgb_dir / f"frame_{t:06d}.png"

        if frame_path.exists():
            from PIL import Image
            import torchvision.transforms.functional as TF
            img = Image.open(frame_path).convert("RGB")
            img = img.resize((self.img_size, self.img_size))
            rgb = TF.to_tensor(img)  # (3, H, W) [0,1]
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb = (rgb - mean) / std
        else:
            # Mock: random normalized RGB
            rgb = torch.randn(3, self.img_size, self.img_size)

        return rgb

    def _load_depth(self, ep_dir: Path, t: int) -> torch.Tensor:
        if self.use_mock_depth:
            return torch.rand(self.img_size, self.img_size) * self.depth_max + 0.1

        depth_dir = ep_dir / "depth"
        depth_path = depth_dir / f"frame_{t:06d}.npy"

        if depth_path.exists():
            depth = np.load(depth_path).astype(np.float32)
            depth = torch.from_numpy(depth)
            # Resize if needed
            if depth.shape != (self.img_size, self.img_size):
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            depth = depth.clamp(0.1, self.depth_max)
        else:
            depth = torch.rand(self.img_size, self.img_size) * self.depth_max + 0.1

        return depth

    def _load_robot_state(self, ep_dir: Path, t: int) -> torch.Tensor:
        states_path = ep_dir / "robot_states.npy"
        if states_path.exists():
            states = np.load(states_path).astype(np.float32)  # (T, d_state)
            state = states[t]
            state = torch.from_numpy(state[:self.d_state])
            # Pad if needed
            if len(state) < self.d_state:
                state = torch.nn.functional.pad(state, (0, self.d_state - len(state)))
        else:
            state = torch.zeros(self.d_state)

        return state

    def _load_action_chunk(self, ep_dir: Path, t: int) -> torch.Tensor:
        actions_path = ep_dir / "actions.npy"
        if actions_path.exists():
            actions = np.load(actions_path).astype(np.float32)  # (T, 7)
            chunk = actions[t : t + self.H]  # (H, 7)
            # Pad if episode too short
            if len(chunk) < self.H:
                pad = np.tile(chunk[-1:], (self.H - len(chunk), 1))
                chunk = np.concatenate([chunk, pad], axis=0)
            return torch.from_numpy(chunk)
        else:
            return torch.zeros(self.H, 7)

    def _load_instruction(self, ep_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        instr_path = ep_dir / "instruction.txt"

        if self.tokenizer is not None and instr_path.exists():
            with open(instr_path) as f:
                instruction = f.read().strip()
            enc = self.tokenizer(
                instruction,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
            )
            return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)
        else:
            # Mock tokenization
            ids  = torch.ones(self.max_len, dtype=torch.long)
            mask = torch.ones(self.max_len, dtype=torch.long)
            return ids, mask

    def _load_camera_K(self, ep_dir: Path) -> Optional[torch.Tensor]:
        K_path = ep_dir / "camera_K.npy"
        if K_path.exists():
            K = np.load(K_path).astype(np.float32)  # (3, 3)
            return torch.from_numpy(K)
        return None


def build_dataloader(
    data_root: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 8,
    **dataset_kwargs,
) -> DataLoader:
    """Build a DataLoader for GST-VLA training."""
    dataset = RobotManipulationDataset(
        data_root=data_root,
        split=split,
        **dataset_kwargs,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    print(f"[DataLoader] {split}: {len(dataset)} samples, {len(loader)} batches")
    return loader


class MockDataLoader:
    """
    Mock DataLoader that generates synthetic batches for quick testing.
    Matches the exact structure expected by GST-VLA.
    """

    def __init__(
        self,
        batch_size: int = 4,
        num_batches: int = 10,
        img_size: int = 224,
        d_state: int = 14,
        H: int = 16,
        d_action: int = 7,
        max_len: int = 64,
    ):
        self.batch_size  = batch_size
        self.num_batches = num_batches
        self.img_size    = img_size
        self.d_state     = d_state
        self.H           = H
        self.d_action    = d_action
        self.max_len     = max_len

    def __iter__(self):
        B = self.batch_size
        for _ in range(self.num_batches):
            yield {
                "rgb":              torch.randn(B, 3, self.img_size, self.img_size),
                "depth_gt":         torch.rand(B, self.img_size, self.img_size) * 4.0 + 0.5,
                "instruction_ids":  torch.ones(B, self.max_len, dtype=torch.long),
                "attention_mask":   torch.ones(B, self.max_len, dtype=torch.long),
                "robot_state":      torch.randn(B, self.d_state),
                "actions":          torch.randn(B, self.H, self.d_action),
                "camera_K":         None,
            }

    def __len__(self):
        return self.num_batches
