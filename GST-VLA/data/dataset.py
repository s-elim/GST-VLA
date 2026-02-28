"""
Dataset for DEAD-VLA Training
==============================
Generic robot manipulation dataset.
Supports LIBERO, MetaWorld, RLBench via the canonical episode format
produced by data_loaders/converters.

Expected directory layout:
    root/
      train_episodes.json   (list of episode dir names)
      val_episodes.json
      episode_000000/
        rgb/           frame_000000.png, ...
        depth/         frame_000000.npy, ... (optional)
        actions.npy         (T, 7) float32
        robot_states.npy    (T, d_state) float32
        instruction.txt
        camera_K.npy        (3, 3) float32  optional
      episode_000001/
      ...

Each sample returns:
    rgb:              (3, 224, 224)  normalized image
    depth_gt:         (224, 224)    metric depth (or mock)
    instruction_ids:  (L,)          tokenized instruction
    attention_mask:   (L,)
    robot_state:      (d_state,)
    actions:          (H, 7)        action chunk
    camera_K:         (3, 3)        optional
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import json


class RobotManipulationDataset(Dataset):
    """Robot manipulation dataset for DEAD-VLA."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        H: int = 16,
        d_state: int = 14,
        img_size: int = 224,
        max_instruction_len: int = 64,
        tokenizer=None,
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
        self.use_mock_depth = use_mock_depth
        self.depth_max = depth_max

        self.episodes = self._load_episode_index()
        self.samples  = self._build_sample_index()

    def _load_episode_index(self) -> List[Path]:
        split_file = self.data_root / f"{self.split}_episodes.json"
        if split_file.exists():
            with open(split_file) as f:
                names = json.load(f)
            return [self.data_root / n for n in names]
        # Fallback: scan directories
        episodes = sorted([d for d in self.data_root.iterdir()
                           if d.is_dir() and d.name.startswith("episode")])
        n = len(episodes)
        return episodes[:int(0.9 * n)] if self.split == "train" else episodes[int(0.9 * n):]

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        samples = []
        for ep_idx, ep_dir in enumerate(self.episodes):
            actions_path = ep_dir / "actions.npy"
            if not actions_path.exists():
                continue
            T = len(np.load(actions_path))
            for t in range(T - self.H):
                samples.append((ep_idx, t))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self.samples[idx]
        ep_dir    = self.episodes[ep_idx]

        rgb              = self._load_rgb(ep_dir, t)
        depth            = self._load_depth(ep_dir, t)
        robot_state      = self._load_robot_state(ep_dir, t)
        actions          = self._load_action_chunk(ep_dir, t)
        inst_ids, mask   = self._load_instruction(ep_dir)
        K                = self._load_camera_K(ep_dir)

        sample = {
            "rgb":             rgb,
            "depth_gt":        depth,
            "instruction_ids": inst_ids,
            "attention_mask":  mask,
            "robot_state":     robot_state,
            "actions":         actions,
        }
        if K is not None:
            sample["camera_K"] = K
        return sample

    def _load_rgb(self, ep_dir: Path, t: int) -> torch.Tensor:
        frame_path = ep_dir / "rgb" / f"frame_{t:06d}.png"
        if frame_path.exists():
            from PIL import Image
            import torchvision.transforms.functional as TF
            img = Image.open(frame_path).convert("RGB").resize((self.img_size, self.img_size))
            rgb = TF.to_tensor(img)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (rgb - mean) / std
        return torch.randn(3, self.img_size, self.img_size)

    def _load_depth(self, ep_dir: Path, t: int) -> torch.Tensor:
        if self.use_mock_depth:
            return torch.rand(self.img_size, self.img_size) * self.depth_max + 0.1
        depth_path = ep_dir / "depth" / f"frame_{t:06d}.npy"
        if depth_path.exists():
            depth = torch.from_numpy(np.load(depth_path).astype(np.float32))
            if depth.shape != (self.img_size, self.img_size):
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=(self.img_size, self.img_size),
                    mode="bilinear", align_corners=False,
                ).squeeze()
            return depth.clamp(0.1, self.depth_max)
        return torch.rand(self.img_size, self.img_size) * self.depth_max + 0.1

    def _load_robot_state(self, ep_dir: Path, t: int) -> torch.Tensor:
        states_path = ep_dir / "robot_states.npy"
        if states_path.exists():
            states = np.load(states_path).astype(np.float32)
            state  = torch.from_numpy(states[t, :self.d_state])
            if len(state) < self.d_state:
                state = torch.nn.functional.pad(state, (0, self.d_state - len(state)))
            return state
        return torch.zeros(self.d_state)

    def _load_action_chunk(self, ep_dir: Path, t: int) -> torch.Tensor:
        actions_path = ep_dir / "actions.npy"
        if actions_path.exists():
            actions = np.load(actions_path).astype(np.float32)
            chunk   = actions[t: t + self.H]
            if len(chunk) < self.H:
                chunk = np.concatenate([chunk, np.tile(chunk[-1:], (self.H - len(chunk), 1))])
            return torch.from_numpy(chunk)
        return torch.zeros(self.H, 7)

    def _load_instruction(self, ep_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        instr_path = ep_dir / "instruction.txt"
        if self.tokenizer is not None and instr_path.exists():
            with open(instr_path) as f:
                instr = f.read().strip()
            enc = self.tokenizer(
                instr, return_tensors="pt", padding="max_length",
                truncation=True, max_length=self.max_len,
            )
            return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)
        return torch.ones(self.max_len, dtype=torch.long), torch.ones(self.max_len, dtype=torch.long)

    def _load_camera_K(self, ep_dir: Path) -> Optional[torch.Tensor]:
        K_path = ep_dir / "camera_K.npy"
        if K_path.exists():
            return torch.from_numpy(np.load(K_path).astype(np.float32))
        return None


def build_dataloader(
    data_root: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 8,
    **dataset_kwargs,
) -> DataLoader:
    dataset = RobotManipulationDataset(data_root=data_root, split=split, **dataset_kwargs)
    loader  = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(split == "train"), num_workers=num_workers,
        pin_memory=True, drop_last=(split == "train"),
    )
    print(f"[DataLoader] {split}: {len(dataset)} samples, {len(loader)} batches")
    return loader


class MockDataLoader:
    """
    Synthetic DataLoader for quick testing without real data.
    Generates random batches matching the exact GST-VLA input format.
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
                "rgb":             torch.randn(B, 3, self.img_size, self.img_size),
                "depth_gt":        torch.rand(B, self.img_size, self.img_size) * 4.0 + 0.5,
                "instruction_ids": torch.ones(B, self.max_len, dtype=torch.long),
                "attention_mask":  torch.ones(B, self.max_len, dtype=torch.long),
                "robot_state":     torch.randn(B, self.d_state),
                "actions":         torch.randn(B, self.H, self.d_action),
                "camera_K":        None,
            }

    def __len__(self):
        return self.num_batches
