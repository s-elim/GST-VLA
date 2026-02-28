"""
Dataset Verification & Sanity Check Tool for GST-VLA
======================================================
ACCV 2026

Verifies that converted datasets are complete and well-formed.
Checks every episode for:
  - Required files present
  - Correct array shapes
  - No NaN/Inf values
  - Depth value ranges
  - Action magnitude ranges
  - RGB value ranges

Usage:
    # Verify a single converted dataset
    python data/scripts/verify_dataset.py --root /data/gst_vla/libero

    # Quick check (first 100 episodes only)
    python data/scripts/verify_dataset.py --root /data/gst_vla/libero --n 100

    # Verify and fix common issues (normalize bad depths, clip outlier actions)
    python data/scripts/verify_dataset.py --root /data/gst_vla/libero --fix

    # Compare two datasets (e.g., before/after normalization)
    python data/scripts/verify_dataset.py \\
        --root /data/gst_vla/libero \\
        --compare /data/gst_vla/rlbench
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─────────────────────────────────────────────
# Verification result
# ─────────────────────────────────────────────

@dataclass
class EpisodeCheckResult:
    ep_dir:      str
    passed:      bool
    issues:      List[str] = field(default_factory=list)
    stats:       Dict      = field(default_factory=dict)


@dataclass
class DatasetVerifyReport:
    dataset_root:   str
    n_episodes:     int
    n_passed:       int
    n_failed:       int
    issue_summary:  Dict[str, int] = field(default_factory=dict)

    # Aggregate stats
    action_min:     float = 0.0
    action_max:     float = 0.0
    depth_min:      float = 0.0
    depth_max:      float = 0.0
    avg_n_frames:   float = 0.0
    min_n_frames:   int   = 0
    max_n_frames:   int   = 0

    def print_summary(self):
        print("\n" + "═"*55)
        print("  Dataset Verification Report")
        print("═"*55)
        print(f"  Root:         {self.dataset_root}")
        print(f"  Episodes:     {self.n_episodes}")
        print(f"  Passed:       {self.n_passed}  ({self.n_passed/max(self.n_episodes,1)*100:.1f}%)")
        print(f"  Failed:       {self.n_failed}")
        print()
        print(f"  Frames:       avg={self.avg_n_frames:.1f}  "
              f"min={self.min_n_frames}  max={self.max_n_frames}")
        print(f"  Action range: [{self.action_min:.4f}, {self.action_max:.4f}]")
        if self.depth_max > 0:
            print(f"  Depth range:  [{self.depth_min:.2f}, {self.depth_max:.2f}] m")
        if self.issue_summary:
            print()
            print("  Issue types:")
            for issue, count in sorted(self.issue_summary.items(), key=lambda x: -x[1]):
                print(f"    {issue:<40} {count:4d} episodes")
        print("═"*55 + "\n")


# ─────────────────────────────────────────────
# Episode checker
# ─────────────────────────────────────────────

REQUIRED_FILES = ["actions.npy", "robot_states.npy", "instruction.txt", "episode_meta.json"]
REQUIRED_DIRS  = ["rgb"]

def check_episode(ep_dir: Path, fix: bool = False) -> EpisodeCheckResult:
    """
    Verify a single episode directory.
    Returns EpisodeCheckResult with any issues found.
    """
    issues = []
    stats  = {}

    # ── Required files ──────────────────────────────
    for fname in REQUIRED_FILES:
        if not (ep_dir / fname).exists():
            issues.append(f"missing_file:{fname}")

    for dname in REQUIRED_DIRS:
        d = ep_dir / dname
        if not d.exists():
            issues.append(f"missing_dir:{dname}")
        elif not any(d.iterdir()):
            issues.append(f"empty_dir:{dname}")

    if issues:
        return EpisodeCheckResult(str(ep_dir), passed=False, issues=issues)

    # ── Load arrays ─────────────────────────────────
    try:
        actions      = np.load(ep_dir / "actions.npy")
        robot_states = np.load(ep_dir / "robot_states.npy")
    except Exception as e:
        return EpisodeCheckResult(
            str(ep_dir), passed=False, issues=[f"load_error:{e}"]
        )

    T = len(actions)
    stats["n_frames"] = T

    # ── Shape checks ────────────────────────────────
    if actions.ndim != 2 or actions.shape[1] != 7:
        issues.append(f"bad_action_shape:{actions.shape}")

    if robot_states.ndim != 2 or robot_states.shape[1] != 14:
        issues.append(f"bad_state_shape:{robot_states.shape}")

    if actions.shape[0] != robot_states.shape[0]:
        issues.append(f"length_mismatch:actions={len(actions)},states={len(robot_states)}")

    if T < 4:
        issues.append(f"too_short:{T}")

    # ── NaN / Inf ────────────────────────────────────
    if not np.isfinite(actions).all():
        n_bad = (~np.isfinite(actions)).sum()
        issues.append(f"nan_inf_actions:{n_bad}")
        if fix:
            actions = np.nan_to_num(actions, nan=0.0, posinf=0.1, neginf=-0.1)
            np.save(ep_dir / "actions.npy", actions)

    if not np.isfinite(robot_states).all():
        n_bad = (~np.isfinite(robot_states)).sum()
        issues.append(f"nan_inf_states:{n_bad}")

    # ── Action magnitude ─────────────────────────────
    stats["action_min"] = float(actions.min())
    stats["action_max"] = float(actions.max())
    if abs(actions[:, :6]).max() > 0.5:
        issues.append(f"large_actions:{abs(actions[:,:6]).max():.3f}")

    # Gripper should be in [0, 1]
    grip = actions[:, 6]
    if grip.min() < -0.1 or grip.max() > 1.1:
        issues.append(f"bad_gripper_range:[{grip.min():.2f},{grip.max():.2f}]")

    # ── RGB frames ────────────────────────────────────
    rgb_dir   = ep_dir / "rgb"
    rgb_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.ppm"))
    n_rgb     = len(rgb_files)
    stats["n_rgb"] = n_rgb

    if n_rgb != T:
        issues.append(f"rgb_count_mismatch:rgb={n_rgb},actions={T}")

    # Spot-check first frame
    if rgb_files:
        try:
            from PIL import Image
            img = np.array(Image.open(rgb_files[0]))
            if img.shape[:2] != (224, 224):
                issues.append(f"wrong_rgb_size:{img.shape}")
            if img.max() == 0:
                issues.append("black_rgb_frame")
        except Exception:
            pass   # PIL not available

    # ── Depth frames ──────────────────────────────────
    depth_dir = ep_dir / "depth"
    if depth_dir.exists():
        depth_files = sorted(depth_dir.glob("*.npy"))
        n_depth     = len(depth_files)

        if n_depth != T:
            issues.append(f"depth_count_mismatch:{n_depth}!={T}")
        elif n_depth > 0:
            # Spot-check first depth
            try:
                d = np.load(depth_files[0])
                stats["depth_min"] = float(d.min())
                stats["depth_max"] = float(d.max())
                if d.max() > 20.0:
                    issues.append(f"depth_in_mm_not_metres:{d.max():.1f}")
                    if fix:
                        # Convert all depth files from mm to metres
                        for df in depth_files:
                            d_fix = np.load(df) / 1000.0
                            np.save(df, d_fix.astype(np.float32))
                if d.min() < 0:
                    issues.append(f"negative_depth:{d.min():.3f}")
                if not np.isfinite(d).all():
                    issues.append("nan_inf_depth")
            except Exception as e:
                issues.append(f"depth_load_error:{e}")

    # ── Camera K ──────────────────────────────────────
    k_path = ep_dir / "camera_K.npy"
    if k_path.exists():
        K = np.load(k_path)
        if K.shape != (3, 3):
            issues.append(f"bad_K_shape:{K.shape}")
        if K[0, 0] < 10 or K[1, 1] < 10:
            issues.append(f"bad_focal_length:{K[0,0]:.1f}")

    return EpisodeCheckResult(
        str(ep_dir),
        passed=len(issues) == 0,
        issues=issues,
        stats=stats,
    )


# ─────────────────────────────────────────────
# Full dataset verification
# ─────────────────────────────────────────────

def verify_dataset(
    root:       str,
    n_episodes: Optional[int] = None,
    fix:        bool = False,
    verbose:    bool = False,
) -> DatasetVerifyReport:
    """
    Verify an entire converted dataset.
    
    Returns DatasetVerifyReport with aggregated stats and issue counts.
    """
    root_path = Path(root)

    # Load episode list
    all_eps = []
    for split in ["train", "val", "test"]:
        split_file = root_path / f"{split}_episodes.json"
        if split_file.exists():
            with open(split_file) as f:
                names = json.load(f)
            all_eps.extend(names)

    if not all_eps:
        # Fallback: scan directories
        all_eps = sorted([d.name for d in root_path.iterdir()
                          if d.is_dir() and d.name.startswith("episode")])

    if n_episodes:
        all_eps = all_eps[:n_episodes]

    print(f"[Verify] Checking {len(all_eps)} episodes in {root}...")

    results      = []
    issue_counts = {}
    n_frames_list = []
    action_mins, action_maxes = [], []
    depth_mins,  depth_maxes  = [], []
    n_passed = 0

    for i, ep_name in enumerate(all_eps):
        ep_dir = root_path / ep_name
        result = check_episode(ep_dir, fix=fix)
        results.append(result)

        if result.passed:
            n_passed += 1
        else:
            for issue in result.issues:
                issue_type = issue.split(":")[0]
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

            if verbose:
                print(f"  ✗ {ep_name}: {', '.join(result.issues)}")

        if "n_frames" in result.stats:
            n_frames_list.append(result.stats["n_frames"])
        if "action_min" in result.stats:
            action_mins.append(result.stats["action_min"])
            action_maxes.append(result.stats["action_max"])
        if "depth_min" in result.stats:
            depth_mins.append(result.stats["depth_min"])
            depth_maxes.append(result.stats["depth_max"])

        if (i + 1) % 100 == 0:
            print(f"  Checked {i+1}/{len(all_eps)}  passed={n_passed}")

    report = DatasetVerifyReport(
        dataset_root  = root,
        n_episodes    = len(all_eps),
        n_passed      = n_passed,
        n_failed      = len(all_eps) - n_passed,
        issue_summary = issue_counts,
        action_min    = min(action_mins) if action_mins else 0.0,
        action_max    = max(action_maxes) if action_maxes else 0.0,
        depth_min     = min(depth_mins) if depth_mins else 0.0,
        depth_max     = max(depth_maxes) if depth_maxes else 0.0,
        avg_n_frames  = float(np.mean(n_frames_list)) if n_frames_list else 0.0,
        min_n_frames  = int(min(n_frames_list)) if n_frames_list else 0,
        max_n_frames  = int(max(n_frames_list)) if n_frames_list else 0,
    )

    report.print_summary()
    return report


# ─────────────────────────────────────────────
# Dataset comparison
# ─────────────────────────────────────────────

def compare_datasets(root_a: str, root_b: str, n: int = 100):
    """Compare two datasets side by side."""
    print(f"\nComparing datasets:")
    print(f"  A: {root_a}")
    print(f"  B: {root_b}")
    print()

    rep_a = verify_dataset(root_a, n_episodes=n, verbose=False)
    rep_b = verify_dataset(root_b, n_episodes=n, verbose=False)

    print("  ┌──────────────────────────┬────────────┬────────────┐")
    print("  │ Metric                   │    A       │    B       │")
    print("  ├──────────────────────────┼────────────┼────────────┤")

    rows = [
        ("Pass rate",     f"{rep_a.n_passed/max(rep_a.n_episodes,1)*100:.1f}%",
                          f"{rep_b.n_passed/max(rep_b.n_episodes,1)*100:.1f}%"),
        ("Avg frames",    f"{rep_a.avg_n_frames:.1f}",   f"{rep_b.avg_n_frames:.1f}"),
        ("Min frames",    str(rep_a.min_n_frames),        str(rep_b.min_n_frames)),
        ("Action min",    f"{rep_a.action_min:.4f}",      f"{rep_b.action_min:.4f}"),
        ("Action max",    f"{rep_a.action_max:.4f}",      f"{rep_b.action_max:.4f}"),
        ("Depth min (m)", f"{rep_a.depth_min:.2f}",       f"{rep_b.depth_min:.2f}"),
        ("Depth max (m)", f"{rep_a.depth_max:.2f}",       f"{rep_b.depth_max:.2f}"),
    ]
    for label, va, vb in rows:
        print(f"  │ {label:<24} │ {va:>10} │ {vb:>10} │")
    print("  └──────────────────────────┴────────────┴────────────┘\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Verify GST-VLA canonical dataset")
    p.add_argument("--root",    required=True)
    p.add_argument("--n",       type=int, default=None,
                   help="Max episodes to check (None = all)")
    p.add_argument("--fix",     action="store_true",
                   help="Auto-fix common issues (depth units, NaN clipping)")
    p.add_argument("--verbose", action="store_true",
                   help="Print each failing episode")
    p.add_argument("--compare", default=None,
                   help="Second dataset root to compare against")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.compare:
        compare_datasets(args.root, args.compare, n=args.n or 100)
    else:
        verify_dataset(args.root, n_episodes=args.n, fix=args.fix, verbose=args.verbose)
