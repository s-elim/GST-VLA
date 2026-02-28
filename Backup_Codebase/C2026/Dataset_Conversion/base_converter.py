"""
Base Dataset Converter for GST-VLA
=====================================
ACCV 2026

Defines the canonical on-disk format that ALL dataset converters
must produce. Every converter (LIBERO, RLBench, MetaWorld, etc.)
inherits from BaseConverter and outputs the same directory structure,
so the training DataLoader in data/dataset.py works with all of them.

──────────────────────────────────────────────────────────────────────
CANONICAL OUTPUT FORMAT
──────────────────────────────────────────────────────────────────────

<output_root>/
  metadata.json              ← dataset-level info
  train_episodes.json        ← list of episode dir names for train split
  val_episodes.json          ← list of episode dir names for val split
  test_episodes.json
  stats.json                 ← normalization stats (action mean/std, etc.)

  episode_000000/
    rgb/
      frame_000000.png       ← (224×224) RGB, uint8
      frame_000001.png
      ...
    depth/
      frame_000000.npy       ← (224×224) float32, metric metres
      frame_000001.npy
      ...
    actions.npy              ← (T, 7)   float32  [Δx,Δy,Δz,Δr,Δp,Δy,grip]
    robot_states.npy         ← (T, 14)  float32  [q(7) + dq(7)]
    camera_K.npy             ← (3, 3)   float32  intrinsics
    instruction.txt          ← plain-text language instruction
    episode_meta.json        ← task_name, success, n_frames, source, etc.

  episode_000001/
  ...

──────────────────────────────────────────────────────────────────────
"""

import os
import json
import shutil
import hashlib
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Canonical episode schema
# ─────────────────────────────────────────────

@dataclass
class EpisodeMeta:
    """Metadata stored per episode in episode_meta.json."""
    episode_id:     str
    task_name:      str
    instruction:    str
    source_dataset: str      # "libero" | "rlbench" | "metaworld" | ...
    source_path:    str      # original file path before conversion
    n_frames:       int
    success:        bool
    robot_type:     str = "panda"      # franka panda default
    camera_name:    str = "front"
    has_depth:      bool = True
    split:          str = "train"
    extra:          Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetStats:
    """Normalization statistics computed over the training set."""
    action_mean:    List[float]      # (7,)
    action_std:     List[float]      # (7,)
    state_mean:     List[float]      # (14,)
    state_std:      List[float]      # (14,)
    depth_mean:     float
    depth_std:      float
    depth_p5:       float
    depth_p95:      float
    n_episodes:     int
    n_frames_total: int
    source:         str


# ─────────────────────────────────────────────
# Base Converter
# ─────────────────────────────────────────────

class BaseConverter(ABC):
    """
    Abstract base class for all GST-VLA dataset converters.

    Subclass and implement:
        - get_episode_list()  → list of raw episode identifiers
        - load_episode()      → raw episode data dict
        - convert_episode()   → EpisodeData (structured)

    Then call:
        converter.run()
    """

    IMG_SIZE   = 224
    D_ACTION   = 7    # [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
    D_STATE    = 14   # [q(7), dq(7)]
    TRAIN_FRAC = 0.9
    VAL_FRAC   = 0.05
    # Remainder goes to test

    def __init__(
        self,
        source_root:  str,
        output_root:  str,
        source_name:  str,
        max_episodes: Optional[int] = None,
        overwrite:    bool = False,
        img_size:     int  = 224,
        depth_max:    float = 5.0,
        verbose:      bool  = True,
    ):
        self.source_root  = Path(source_root)
        self.output_root  = Path(output_root)
        self.source_name  = source_name
        self.max_episodes = max_episodes
        self.overwrite    = overwrite
        self.img_size     = img_size
        self.depth_max    = depth_max
        self.verbose      = verbose
        self._episode_counter = 0

        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="[%(levelname)s] %(message)s",
        )

    # ── Abstract interface ─────────────────────────────────

    @abstractmethod
    def get_episode_list(self) -> List[Any]:
        """Return list of raw episode identifiers (paths, keys, etc.)."""
        ...

    @abstractmethod
    def load_raw_episode(self, episode_id: Any) -> Dict[str, Any]:
        """
        Load one episode from the source dataset.
        Returns a dict with raw (unconverted) data.
        Keys depend on source format — handled in convert_episode.
        """
        ...

    @abstractmethod
    def convert_episode(
        self,
        raw: Dict[str, Any],
        episode_id: Any,
    ) -> Optional["EpisodeData"]:
        """
        Convert raw episode data to canonical EpisodeData.
        Return None to skip this episode.
        """
        ...

    # ── Canonical output writer ────────────────────────────

    def write_episode(self, ep_data: "EpisodeData", ep_dir: Path):
        """Write one canonical episode to disk."""
        ep_dir.mkdir(parents=True, exist_ok=True)

        # RGB frames
        rgb_dir = ep_dir / "rgb"
        rgb_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(ep_data.rgb_frames):
            _save_rgb(frame, rgb_dir / f"frame_{i:06d}.png")

        # Depth frames
        if ep_data.depth_frames:
            depth_dir = ep_dir / "depth"
            depth_dir.mkdir(exist_ok=True)
            for i, d in enumerate(ep_data.depth_frames):
                np.save(depth_dir / f"frame_{i:06d}.npy", d.astype(np.float32))

        # Actions
        np.save(ep_dir / "actions.npy",      ep_data.actions.astype(np.float32))
        np.save(ep_dir / "robot_states.npy", ep_data.robot_states.astype(np.float32))

        # Camera K
        if ep_data.camera_K is not None:
            np.save(ep_dir / "camera_K.npy", ep_data.camera_K.astype(np.float32))

        # Instruction
        (ep_dir / "instruction.txt").write_text(ep_data.meta.instruction)

        # Metadata
        with open(ep_dir / "episode_meta.json", "w") as f:
            json.dump(asdict(ep_data.meta), f, indent=2)

    def run(self) -> DatasetStats:
        """
        Main conversion loop.
        Processes all episodes, writes canonical format, computes stats.
        """
        self.output_root.mkdir(parents=True, exist_ok=True)

        episodes = self.get_episode_list()
        if self.max_episodes:
            episodes = episodes[:self.max_episodes]

        logger.info(f"[{self.source_name}] Found {len(episodes)} episodes")

        # Split
        n      = len(episodes)
        n_train = int(n * self.TRAIN_FRAC)
        n_val   = int(n * self.VAL_FRAC)
        splits  = (
            ["train"] * n_train +
            ["val"]   * n_val +
            ["test"]  * (n - n_train - n_val)
        )

        train_names, val_names, test_names = [], [], []
        all_actions, all_states, all_depths = [], [], []
        n_frames_total = 0
        n_skipped = 0

        for i, (ep_id, split) in enumerate(zip(episodes, splits)):
            ep_dir_name = f"episode_{self._episode_counter:06d}"
            ep_dir      = self.output_root / ep_dir_name

            if ep_dir.exists() and not self.overwrite:
                logger.debug(f"  Skip (exists): {ep_dir_name}")
                self._episode_counter += 1
                if split == "train": train_names.append(ep_dir_name)
                elif split == "val": val_names.append(ep_dir_name)
                else:                test_names.append(ep_dir_name)
                continue

            try:
                raw     = self.load_raw_episode(ep_id)
                ep_data = self.convert_episode(raw, ep_id)
                if ep_data is None:
                    n_skipped += 1
                    continue

                ep_data.meta.split      = split
                ep_data.meta.episode_id = ep_dir_name

                self.write_episode(ep_data, ep_dir)

                # Accumulate stats
                all_actions.append(ep_data.actions)
                all_states.append(ep_data.robot_states)
                if ep_data.depth_frames:
                    d_flat = np.concatenate([d.flatten() for d in ep_data.depth_frames])
                    all_depths.append(d_flat)
                n_frames_total += len(ep_data.rgb_frames)

            except Exception as e:
                logger.warning(f"  Episode {ep_id} failed: {e}")
                n_skipped += 1
                continue

            if split == "train": train_names.append(ep_dir_name)
            elif split == "val": val_names.append(ep_dir_name)
            else:                test_names.append(ep_dir_name)

            self._episode_counter += 1

            if self.verbose and (i + 1) % 50 == 0:
                logger.info(f"  Converted {i+1}/{len(episodes)}  (skipped={n_skipped})")

        # Write split files
        for fname, names in [
            ("train_episodes.json", train_names),
            ("val_episodes.json",   val_names),
            ("test_episodes.json",  test_names),
        ]:
            with open(self.output_root / fname, "w") as f:
                json.dump(names, f, indent=2)

        # Compute normalization stats
        stats = self._compute_stats(
            all_actions, all_states, all_depths,
            n_frames_total,
            n_episodes=len(train_names) + len(val_names) + len(test_names),
        )

        with open(self.output_root / "stats.json", "w") as f:
            json.dump(asdict(stats), f, indent=2)

        # Dataset metadata
        meta = {
            "source":        self.source_name,
            "n_train":       len(train_names),
            "n_val":         len(val_names),
            "n_test":        len(test_names),
            "n_skipped":     n_skipped,
            "n_frames_total":n_frames_total,
            "img_size":      self.img_size,
            "d_action":      self.D_ACTION,
            "d_state":       self.D_STATE,
        }
        with open(self.output_root / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"\n[{self.source_name}] Conversion complete!")
        logger.info(f"  Train: {len(train_names)}  Val: {len(val_names)}  "
                    f"Test: {len(test_names)}  Skipped: {n_skipped}")
        logger.info(f"  Total frames: {n_frames_total}")
        logger.info(f"  Output: {self.output_root}")

        return stats

    def _compute_stats(
        self,
        all_actions: List[np.ndarray],
        all_states:  List[np.ndarray],
        all_depths:  List[np.ndarray],
        n_frames:    int,
        n_episodes:  int,
    ) -> DatasetStats:
        """Compute normalization statistics over training set."""
        if all_actions:
            acts = np.concatenate(all_actions, axis=0)  # (N, 7)
            a_mean = acts.mean(axis=0).tolist()
            a_std  = (acts.std(axis=0) + 1e-6).tolist()
        else:
            a_mean = [0.0] * self.D_ACTION
            a_std  = [1.0] * self.D_ACTION

        if all_states:
            sts = np.concatenate(all_states, axis=0)
            s_mean = sts.mean(axis=0).tolist()
            s_std  = (sts.std(axis=0) + 1e-6).tolist()
        else:
            s_mean = [0.0] * self.D_STATE
            s_std  = [1.0] * self.D_STATE

        if all_depths:
            depths = np.concatenate(all_depths)
            d_mean = float(depths.mean())
            d_std  = float(depths.std() + 1e-6)
            d_p5   = float(np.percentile(depths, 5))
            d_p95  = float(np.percentile(depths, 95))
        else:
            d_mean, d_std, d_p5, d_p95 = 1.5, 0.8, 0.3, 3.5

        return DatasetStats(
            action_mean=a_mean, action_std=a_std,
            state_mean=s_mean,  state_std=s_std,
            depth_mean=d_mean,  depth_std=d_std,
            depth_p5=d_p5,      depth_p95=d_p95,
            n_episodes=n_episodes,
            n_frames_total=n_frames,
            source=self.source_name,
        )


# ─────────────────────────────────────────────
# EpisodeData container
# ─────────────────────────────────────────────

@dataclass
class EpisodeData:
    """Canonical episode data ready to write to disk."""
    meta:          EpisodeMeta
    rgb_frames:    List[np.ndarray]            # list of (H, W, 3) uint8
    depth_frames:  Optional[List[np.ndarray]]  # list of (H, W) float32 metres
    actions:       np.ndarray                  # (T, 7) float32
    robot_states:  np.ndarray                  # (T, 14) float32
    camera_K:      Optional[np.ndarray]        # (3, 3) float32


# ─────────────────────────────────────────────
# Shared image/depth utils
# ─────────────────────────────────────────────

def _save_rgb(frame: np.ndarray, path: Path):
    """Save (H, W, 3) uint8 RGB as PNG."""
    try:
        from PIL import Image
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(frame).save(str(path))
    except ImportError:
        # Fallback: write raw bytes header (PPM)
        h, w = frame.shape[:2]
        with open(str(path).replace(".png", ".ppm"), "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(frame.tobytes())


def resize_rgb(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize HxWx3 uint8 to size×size."""
    try:
        from PIL import Image
        img = Image.fromarray(frame.astype(np.uint8))
        return np.array(img.resize((size, size), Image.BILINEAR))
    except ImportError:
        # Basic nearest-neighbour fallback (no PIL)
        h, w = frame.shape[:2]
        if h == size and w == size:
            return frame
        y_idx = (np.arange(size) * h / size).astype(int)
        x_idx = (np.arange(size) * w / size).astype(int)
        return frame[np.ix_(y_idx, x_idx)]


def resize_depth(depth: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize HxW float32 depth to size×size."""
    try:
        from PIL import Image
        img = Image.fromarray(depth.astype(np.float32), mode="F")
        return np.array(img.resize((size, size), Image.BILINEAR))
    except ImportError:
        h, w = depth.shape
        if h == size and w == size:
            return depth
        y_idx = (np.arange(size) * h / size).astype(int)
        x_idx = (np.arange(size) * w / size).astype(int)
        return depth[np.ix_(y_idx, x_idx)]


def build_K_from_fov(h: int, w: int, fov_deg: float = 60.0) -> np.ndarray:
    """Compute 3×3 pinhole intrinsic matrix from field-of-view."""
    import math
    fx = w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    fy = fx
    return np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float32)


def eef_to_delta_actions(
    eef_poses: np.ndarray,  # (T, 6) [x,y,z,r,p,y] absolute
    gripper:   np.ndarray,  # (T,)   absolute gripper state
    clip_val:  float = 0.1,
) -> np.ndarray:
    """
    Convert absolute EEF poses to delta actions.
    
    Δpose_t = pose_{t+1} - pose_t
    Last frame duplicated to maintain length T.
    
    Returns:
        actions: (T, 7) [Δx,Δy,Δz,Δr,Δp,Δyaw, grip]
    """
    T = len(eef_poses)
    delta_pose = np.zeros((T, 6), dtype=np.float32)
    delta_pose[:-1] = eef_poses[1:] - eef_poses[:-1]
    delta_pose[-1]  = delta_pose[-2]  # repeat last

    # Clip large deltas
    delta_pose = np.clip(delta_pose, -clip_val, clip_val)

    # Gripper: normalize to [0, 1] and discretize
    grip_norm = gripper.astype(np.float32)
    if grip_norm.max() > 1.0:
        grip_norm = grip_norm / grip_norm.max()
    grip_norm = grip_norm.reshape(-1, 1)

    return np.concatenate([delta_pose, grip_norm], axis=-1)  # (T, 7)
