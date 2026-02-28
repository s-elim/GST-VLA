"""
Dataset Normalization & Statistics Utilities for GST-VLA
==========================================================
ACCV 2026

Loads per-dataset normalization stats computed during conversion,
and provides normalizers/denormalizers used in the training DataLoader
and at inference time.

Usage:
    stats = load_dataset_stats("/data/gst_vla/libero/stats.json")
    normalizer = ActionNormalizer(stats)
    
    # In DataLoader
    actions_norm = normalizer.normalize(actions)  # → ~N(0,1)
    
    # At inference (unnormalize model output)
    actions_raw = normalizer.denormalize(pred_actions_norm)
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass


# ─────────────────────────────────────────────
# Stats loading
# ─────────────────────────────────────────────

@dataclass
class NormStats:
    action_mean:  np.ndarray    # (7,)
    action_std:   np.ndarray    # (7,)
    state_mean:   np.ndarray    # (14,)
    state_std:    np.ndarray    # (14,)
    depth_mean:   float
    depth_std:    float
    depth_p5:     float
    depth_p95:    float
    source:       str


def load_dataset_stats(stats_path: Union[str, Path]) -> NormStats:
    """Load normalization stats from stats.json file."""
    with open(stats_path) as f:
        data = json.load(f)
    return NormStats(
        action_mean = np.array(data["action_mean"], dtype=np.float32),
        action_std  = np.array(data["action_std"],  dtype=np.float32),
        state_mean  = np.array(data["state_mean"],  dtype=np.float32),
        state_std   = np.array(data["state_std"],   dtype=np.float32),
        depth_mean  = float(data["depth_mean"]),
        depth_std   = float(data["depth_std"]),
        depth_p5    = float(data["depth_p5"]),
        depth_p95   = float(data["depth_p95"]),
        source      = data.get("source", "unknown"),
    )


def merge_dataset_stats(stats_list: List[NormStats]) -> NormStats:
    """
    Merge normalization stats from multiple datasets into one combined stats.
    Uses weighted mean of means and combined std estimate.
    Used when training on LIBERO + RLBench + MetaWorld jointly.
    """
    # Simple mean of means/stds (approximate but sufficient)
    a_means = np.stack([s.action_mean for s in stats_list])
    a_stds  = np.stack([s.action_std  for s in stats_list])
    s_means = np.stack([s.state_mean  for s in stats_list])
    s_stds  = np.stack([s.state_std   for s in stats_list])

    return NormStats(
        action_mean = a_means.mean(axis=0),
        action_std  = np.sqrt((a_stds**2).mean(axis=0)) + 1e-6,
        state_mean  = s_means.mean(axis=0),
        state_std   = np.sqrt((s_stds**2).mean(axis=0)) + 1e-6,
        depth_mean  = np.mean([s.depth_mean for s in stats_list]),
        depth_std   = np.mean([s.depth_std  for s in stats_list]),
        depth_p5    = np.min([s.depth_p5    for s in stats_list]),
        depth_p95   = np.max([s.depth_p95   for s in stats_list]),
        source      = "+".join(set(s.source for s in stats_list)),
    )


# ─────────────────────────────────────────────
# Normalizers
# ─────────────────────────────────────────────

class ActionNormalizer:
    """
    Normalizes and denormalizes 7-DoF action vectors.
    
    normalize:   (T, 7) raw → (T, 7) ~N(0,1)
    denormalize: (T, 7) ~N(0,1) → (T, 7) raw (for model output at inference)
    """

    def __init__(self, stats: NormStats):
        self.mean = stats.action_mean   # (7,)
        self.std  = stats.action_std    # (7,)

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        """(*, 7) → (*, 7)"""
        return (actions - self.mean) / (self.std + 1e-6)

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        """(*, 7) → (*, 7)"""
        return actions * self.std + self.mean


class StateNormalizer:
    """Normalizes 14-DoF robot state vectors."""

    def __init__(self, stats: NormStats):
        self.mean = stats.state_mean
        self.std  = stats.state_std

    def normalize(self, state: np.ndarray) -> np.ndarray:
        return (state - self.mean) / (self.std + 1e-6)

    def denormalize(self, state: np.ndarray) -> np.ndarray:
        return state * self.std + self.mean


class DepthNormalizer:
    """
    Normalizes depth maps.
    Supports two modes:
      'zscore':  (d - mean) / std
      'percentile': clip to [p5, p95] then scale to [0, 1]
    """

    def __init__(self, stats: NormStats, mode: str = "percentile"):
        self.mean  = stats.depth_mean
        self.std   = stats.depth_std
        self.p5    = stats.depth_p5
        self.p95   = stats.depth_p95
        self.mode  = mode

    def normalize(self, depth: np.ndarray) -> np.ndarray:
        """(H, W) or (B, H, W) → normalized"""
        if self.mode == "zscore":
            return (depth - self.mean) / (self.std + 1e-6)
        # percentile
        d_clip = np.clip(depth, self.p5, self.p95)
        return (d_clip - self.p5) / (self.p95 - self.p5 + 1e-6)

    def denormalize(self, depth_norm: np.ndarray) -> np.ndarray:
        if self.mode == "zscore":
            return depth_norm * self.std + self.mean
        return depth_norm * (self.p95 - self.p5) + self.p5


# ─────────────────────────────────────────────
# Multi-Dataset Loader (combines converted datasets)
# ─────────────────────────────────────────────

class MultiDatasetConfig:
    """
    Configuration for loading and mixing multiple canonical datasets.
    
    Handles:
      - Weighted sampling across datasets
      - Shared normalization stats
      - Split file merging
    """

    def __init__(
        self,
        dataset_roots: Dict[str, str],    # {"libero": "/data/libero", ...}
        weights: Optional[Dict[str, float]] = None,
        split: str = "train",
    ):
        self.dataset_roots = {k: Path(v) for k, v in dataset_roots.items()}
        self.weights = weights or {k: 1.0 for k in dataset_roots}
        self.split   = split

        self._stats: Optional[NormStats] = None
        self._episode_index: Optional[List[Path]] = None

    def get_stats(self) -> NormStats:
        """Load and merge normalization stats from all datasets."""
        if self._stats is None:
            stats_list = []
            for name, root in self.dataset_roots.items():
                stats_path = root / "stats.json"
                if stats_path.exists():
                    stats_list.append(load_dataset_stats(stats_path))
                else:
                    print(f"  [Warning] No stats.json for {name}, using defaults")
            if not stats_list:
                # fallback
                self._stats = _default_stats()
            elif len(stats_list) == 1:
                self._stats = stats_list[0]
            else:
                self._stats = merge_dataset_stats(stats_list)
        return self._stats

    def get_episode_list(self) -> List[Path]:
        """Build weighted episode list across all datasets."""
        if self._episode_index is not None:
            return self._episode_index

        all_episodes = []

        for name, root in self.dataset_roots.items():
            split_file = root / f"{self.split}_episodes.json"
            if not split_file.exists():
                print(f"  [Warning] No {self.split}_episodes.json for {name}")
                continue

            with open(split_file) as f:
                ep_names = json.load(f)

            ep_paths = [root / ep for ep in ep_names]
            weight = self.weights.get(name, 1.0)

            # Replicate by weight (integer rounding)
            n_reps = max(1, round(weight))
            weighted = ep_paths * n_reps
            all_episodes.extend(weighted)
            print(f"  [{name}] {len(ep_paths)} episodes × {n_reps} = {len(weighted)}")

        # Shuffle
        np.random.shuffle(all_episodes)
        self._episode_index = all_episodes
        print(f"  Total (weighted): {len(all_episodes)} episodes")
        return all_episodes


def _default_stats() -> NormStats:
    """Return sensible default normalization stats."""
    return NormStats(
        action_mean = np.zeros(7,  dtype=np.float32),
        action_std  = np.ones(7,   dtype=np.float32) * 0.05,
        state_mean  = np.zeros(14, dtype=np.float32),
        state_std   = np.ones(14,  dtype=np.float32),
        depth_mean  = 1.5,
        depth_std   = 0.8,
        depth_p5    = 0.3,
        depth_p95   = 3.5,
        source      = "default",
    )
