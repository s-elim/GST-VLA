"""
LIBERO Dataset Converter for GST-VLA
======================================
ACCV 2026

Converts LIBERO benchmark datasets to the canonical GST-VLA format.

LIBERO Structure:
    LIBERO provides HDF5 files organised by task suite:
        libero_spatial/   → 10 tasks, spatial reasoning
        libero_object/    → 10 tasks, object manipulation
        libero_goal/      → 10 tasks, goal-conditioned
        libero_long/      → 10 tasks, long-horizon
        libero_90/        → 90 tasks, large-scale

    Each HDF5 file:
        data/demo_N/
            obs/
                agentview_rgb          (T, 3, H, W) uint8
                robot0_eye_in_hand_rgb (T, 3, H, W) optional
                agentview_depth        (T, H, W)    optional float32
                robot0_eef_pos         (T, 3)
                robot0_eef_quat        (T, 4)
                robot0_gripper_qpos    (T, 2)
                robot0_joint_pos       (T, 7)
                robot0_joint_vel       (T, 7)
            actions                   (T, 7)  absolute or delta
            rewards                   (T,)
            dones                     (T,)

Usage:
    python data/scripts/convert_libero.py \\
        --source /data/libero \\
        --output /data/gst_vla/libero \\
        --suites libero_spatial libero_object libero_goal libero_long \\
        --max_episodes 200

    # Or programmatically:
    converter = LIBEROConverter(
        source_root="/data/libero",
        output_root="/data/gst_vla/libero",
        suites=["libero_spatial", "libero_object"],
    )
    stats = converter.run()
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from data.converters.base_converter import (
    BaseConverter, EpisodeData, EpisodeMeta,
    resize_rgb, resize_depth, build_K_from_fov, eef_to_delta_actions,
)

import logging
logger = logging.getLogger(__name__)


# ── LIBERO camera intrinsics (from official config) ──────────────────
# agentview camera: 128×128 with ~45° FOV in official releases,
# but most labs use 256×256 or 224×224 — we normalise at conversion.
LIBERO_CAMERA_FOV  = 45.0       # degrees, agentview
LIBERO_IMG_SIZE_H  = 128        # native resolution in most LIBERO HDF5
LIBERO_IMG_SIZE_W  = 128

# Map task suite folder names → short tag for instructions
SUITE_TAGS = {
    "libero_spatial": "spatial",
    "libero_object":  "object",
    "libero_goal":    "goal",
    "libero_long":    "long",
    "libero_90":      "90",
}


class LIBEROConverter(BaseConverter):
    """
    Converts LIBERO benchmark HDF5 files → GST-VLA canonical format.

    Args:
        source_root:    root containing libero_spatial/, libero_object/, ...
        output_root:    where to write canonical episodes
        suites:         which LIBERO suites to include (None = all found)
        camera:         which camera to use ('agentview' | 'eye_in_hand')
        use_depth:      whether to load depth channel (if available)
        action_type:    'delta' (convert abs→delta) | 'raw' (use as-is)
    """

    def __init__(
        self,
        source_root:   str,
        output_root:   str,
        suites:        Optional[List[str]] = None,
        camera:        str   = "agentview",
        use_depth:     bool  = True,
        action_type:   str   = "delta",
        max_episodes:  Optional[int] = None,
        overwrite:     bool  = False,
        img_size:      int   = 224,
        verbose:       bool  = True,
    ):
        super().__init__(
            source_root=source_root,
            output_root=output_root,
            source_name="libero",
            max_episodes=max_episodes,
            overwrite=overwrite,
            img_size=img_size,
            verbose=verbose,
        )
        self.suites      = suites
        self.camera      = camera
        self.use_depth   = use_depth
        self.action_type = action_type

    # ─────────────────────────────────────────
    # Episode discovery
    # ─────────────────────────────────────────

    def get_episode_list(self) -> List[Tuple[Path, str, int]]:
        """
        Returns list of (hdf5_path, suite_name, demo_idx) tuples.
        """
        episodes = []
        source = self.source_root

        available_suites = sorted([
            d.name for d in Path(source).iterdir()
            if d.is_dir() and (self.suites is None or d.name in self.suites)
        ])

        if not available_suites:
            # Try treating source_root itself as a suite with HDF5 files
            available_suites = ["."]

        for suite_name in available_suites:
            suite_dir = Path(source) / suite_name
            hdf5_files = sorted(suite_dir.glob("*.hdf5")) + sorted(suite_dir.glob("*.h5"))
            for hdf5_path in hdf5_files:
                n_demos = self._count_demos(hdf5_path)
                for i in range(n_demos):
                    episodes.append((hdf5_path, suite_name, i))
                logger.info(f"  [{suite_name}] {hdf5_path.name}: {n_demos} demos")

        logger.info(f"[LIBERO] Total episodes found: {len(episodes)}")
        return episodes

    def _count_demos(self, hdf5_path: Path) -> int:
        """Count number of demos in an HDF5 file."""
        try:
            import h5py
            with h5py.File(str(hdf5_path), "r") as f:
                return len(f["data"].keys())
        except Exception as e:
            logger.warning(f"Could not count demos in {hdf5_path}: {e}")
            return 0

    # ─────────────────────────────────────────
    # Raw episode loading
    # ─────────────────────────────────────────

    def load_raw_episode(self, episode_id: Tuple[Path, str, int]) -> Dict[str, Any]:
        """Load raw HDF5 data for one demo."""
        hdf5_path, suite_name, demo_idx = episode_id

        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required: pip install h5py")

        with h5py.File(str(hdf5_path), "r") as f:
            # Get demo key
            demo_keys = sorted(f["data"].keys(),
                               key=lambda k: int(k.replace("demo_", "")))
            if demo_idx >= len(demo_keys):
                return {}
            demo_key = demo_keys[demo_idx]
            demo     = f["data"][demo_key]

            raw = {
                "suite_name":  suite_name,
                "hdf5_path":   str(hdf5_path),
                "demo_key":    demo_key,
                "task_name":   hdf5_path.stem,   # e.g. "pick_up_the_alphabet_soup"
            }

            obs = demo["obs"]

            # ── RGB ──────────────────────────────────
            rgb_key = f"{self.camera}_rgb"
            if rgb_key in obs:
                raw["rgb"] = obs[rgb_key][:]       # (T, 3, H, W) or (T, H, W, 3)
            elif "agentview_rgb" in obs:
                raw["rgb"] = obs["agentview_rgb"][:]
            else:
                # Try to find any rgb key
                rgb_keys = [k for k in obs.keys() if "rgb" in k.lower()]
                raw["rgb"] = obs[rgb_keys[0]][:] if rgb_keys else None

            # ── Depth ─────────────────────────────────
            depth_key = f"{self.camera}_depth"
            if self.use_depth and depth_key in obs:
                raw["depth"] = obs[depth_key][:]   # (T, H, W) float32 in metres
            else:
                raw["depth"] = None

            # ── Robot state ───────────────────────────
            raw["joint_pos"] = obs["robot0_joint_pos"][:] if "robot0_joint_pos" in obs \
                               else np.zeros((len(raw["rgb"]), 7))
            raw["joint_vel"] = obs["robot0_joint_vel"][:] if "robot0_joint_vel" in obs \
                               else np.zeros((len(raw["rgb"]), 7))
            raw["eef_pos"]   = obs["robot0_eef_pos"][:]   if "robot0_eef_pos"   in obs \
                               else np.zeros((len(raw["rgb"]), 3))
            raw["eef_quat"]  = obs["robot0_eef_quat"][:]  if "robot0_eef_quat"  in obs \
                               else np.zeros((len(raw["rgb"]), 4))
            raw["gripper"]   = obs["robot0_gripper_qpos"][:] if "robot0_gripper_qpos" in obs \
                               else np.zeros((len(raw["rgb"]), 2))

            # ── Actions ───────────────────────────────
            raw["actions"] = demo["actions"][:] if "actions" in demo else None

            # ── Success / reward ──────────────────────
            raw["dones"]   = demo["dones"][:] if "dones" in demo else None
            raw["rewards"] = demo["rewards"][:] if "rewards" in demo else None

            # ── Language instruction ──────────────────
            # LIBERO stores task instructions in metadata or as attr
            if hasattr(demo, "attrs") and "model_file" in demo.attrs:
                raw["instruction"] = self._infer_instruction(raw["task_name"])
            elif "language_instruction" in obs:
                raw["instruction"] = str(obs["language_instruction"][0])
            else:
                raw["instruction"] = self._infer_instruction(raw["task_name"])

        return raw

    def _infer_instruction(self, task_name: str) -> str:
        """Convert task filename to natural language instruction."""
        # E.g. "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
        #    → "Pick up the alphabet soup and place it in the basket"
        instruction = task_name.replace("_", " ").strip()
        return instruction.capitalize() + "."

    # ─────────────────────────────────────────
    # Episode conversion
    # ─────────────────────────────────────────

    def convert_episode(
        self,
        raw: Dict[str, Any],
        episode_id: Any,
    ) -> Optional[EpisodeData]:
        """Convert raw LIBERO episode to canonical EpisodeData."""
        if not raw or raw.get("rgb") is None:
            return None

        rgb_raw = raw["rgb"]   # (T, 3, H, W) uint8

        # Ensure (T, H, W, 3) layout
        if rgb_raw.ndim == 4 and rgb_raw.shape[1] == 3:
            rgb_raw = rgb_raw.transpose(0, 2, 3, 1)  # (T, H, W, 3)
        T, H, W, _ = rgb_raw.shape

        if T < 2:
            return None

        # ── RGB frames ────────────────────────────────
        rgb_frames = [resize_rgb(rgb_raw[t], self.img_size) for t in range(T)]

        # ── Depth frames ──────────────────────────────
        depth_frames = None
        if raw.get("depth") is not None:
            depth_raw = raw["depth"]  # (T, H, W) float32
            depth_frames = [
                resize_depth(depth_raw[t], self.img_size).clip(0.01, self.depth_max)
                for t in range(T)
            ]

        # ── Robot state [q(7), dq(7)] ─────────────────
        joint_pos = raw["joint_pos"].astype(np.float32)[:T]   # (T, 7)
        joint_vel = raw["joint_vel"].astype(np.float32)[:T]   # (T, 7)

        # Pad to D_STATE=14 if joint dims differ
        if joint_pos.shape[1] < 7:
            joint_pos = np.pad(joint_pos, ((0,0),(0,7-joint_pos.shape[1])))
        if joint_vel.shape[1] < 7:
            joint_vel = np.pad(joint_vel, ((0,0),(0,7-joint_vel.shape[1])))

        robot_states = np.concatenate([joint_pos[:,:7], joint_vel[:,:7]], axis=-1)  # (T, 14)

        # ── Actions ───────────────────────────────────
        if raw.get("actions") is not None and self.action_type == "raw":
            # Use stored actions directly (already delta in some LIBERO versions)
            actions_raw = raw["actions"].astype(np.float32)[:T]
            if actions_raw.shape[1] >= 7:
                actions = actions_raw[:, :7]
            else:
                # Pad gripper if missing
                actions = np.pad(actions_raw, ((0,0),(0,7-actions_raw.shape[1])))
        else:
            # Compute delta from EEF poses
            eef_pos  = raw["eef_pos"].astype(np.float32)[:T]   # (T, 3)
            eef_quat = raw["eef_quat"].astype(np.float32)[:T]  # (T, 4) xyzw

            # Convert quaternion to euler angles for delta computation
            eef_euler = _quat_to_euler(eef_quat)                # (T, 3)
            eef_6dof  = np.concatenate([eef_pos, eef_euler], axis=-1)  # (T, 6)

            gripper_raw = raw["gripper"].astype(np.float32)[:T]
            gripper_state = gripper_raw.mean(axis=-1) if gripper_raw.ndim == 2 \
                            else gripper_raw   # (T,)

            actions = eef_to_delta_actions(eef_6dof, gripper_state)  # (T, 7)

        # ── Camera intrinsics ──────────────────────────
        camera_K = build_K_from_fov(self.img_size, self.img_size, LIBERO_CAMERA_FOV)

        # ── Success ────────────────────────────────────
        success = False
        if raw.get("dones") is not None:
            success = bool(raw["dones"][-1]) or bool(raw["dones"].max())
        elif raw.get("rewards") is not None:
            success = float(raw["rewards"][-1]) > 0.5

        # ── Metadata ──────────────────────────────────
        hdf5_path, suite_name, demo_idx = episode_id
        meta = EpisodeMeta(
            episode_id     = "",  # filled by write_episode
            task_name      = raw["task_name"],
            instruction    = raw["instruction"],
            source_dataset = "libero",
            source_path    = f"{hdf5_path}::demo_{demo_idx}",
            n_frames       = T,
            success        = success,
            robot_type     = "panda",
            camera_name    = self.camera,
            has_depth      = depth_frames is not None,
            extra          = {"suite": suite_name, "demo_key": raw["demo_key"]},
        )

        return EpisodeData(
            meta=meta,
            rgb_frames=rgb_frames,
            depth_frames=depth_frames,
            actions=actions,
            robot_states=robot_states,
            camera_K=camera_K,
        )


# ─────────────────────────────────────────────
# Quaternion → Euler (ZYX / roll-pitch-yaw)
# ─────────────────────────────────────────────

def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert (T, 4) xyzw quaternions to (T, 3) roll-pitch-yaw euler angles.
    Uses ZYX convention (same as ROS).
    """
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Roll (x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

    # Yaw (z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)
