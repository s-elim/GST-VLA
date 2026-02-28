"""
MetaWorld Dataset Converter for GST-VLA
=========================================
ACCV 2026

Converts MetaWorld rollout data → canonical GST-VLA format.

MetaWorld provides a gym-style API. We support two input modes:

Mode 1: Pre-recorded HDF5 (from collect_demos.py)
    demos/
      metaworld_mt10/
        assembly-v2/
          demos.hdf5
        basketball-v2/
          demos.hdf5
        ...

Mode 2: Live collection via MetaWorld env (requires metaworld package)
    Used when hdf5_root is not provided.
    Runs scripted policy for N episodes per task.

Supported tasks (MT10, MT50):
    MT10: reach, push, pick-place, door-open, drawer-open, drawer-close,
          button-press-topdown, peg-insert-side, window-open, window-close
    MT50: all above + 40 more tasks

HDF5 format (from collect_demos.py):
    data/
      traj_0/
        obs/
          image          (T, 3, 64, 64) or (T, 64, 64, 3) uint8
          state          (T, 39)  [robot(18) + obj(21)]
        actions          (T, 4)   [Δx,Δy,Δz, grip]
        rewards          (T,)
        terminals        (T,)
        language_goal    string

Usage:
    python data/scripts/convert_metaworld.py \\
        --source /data/metaworld_demos \\
        --output /data/gst_vla/metaworld \\
        --tasks reach-v2 push-v2 pick-place-v2

    # Live collection mode (requires metaworld installed):
    python data/scripts/convert_metaworld.py \\
        --live --tasks MT10 --n_demos 50 \\
        --output /data/gst_vla/metaworld
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
from data.converters.libero_converter import _quat_to_euler

import logging
logger = logging.getLogger(__name__)


# MetaWorld camera FOV (MuJoCo default for offscreen render)
METAWORLD_CAMERA_FOV  = 55.0
METAWORLD_NATIVE_SIZE = 84    # most pre-rendered demos use 84×84

MT10_TASKS = [
    "reach-v2", "push-v2", "pick-place-v2", "door-open-v2",
    "drawer-open-v2", "drawer-close-v2", "button-press-topdown-v2",
    "peg-insert-side-v2", "window-open-v2", "window-close-v2",
]

# Language instructions for MetaWorld tasks
METAWORLD_INSTRUCTIONS = {
    "reach-v2":                  "Reach the goal position with the end-effector.",
    "push-v2":                   "Push the puck to the goal position.",
    "pick-place-v2":             "Pick up the object and place it at the goal.",
    "door-open-v2":              "Open the door by rotating the handle.",
    "drawer-open-v2":            "Pull the drawer open.",
    "drawer-close-v2":           "Push the drawer closed.",
    "button-press-topdown-v2":   "Press the button from above.",
    "peg-insert-side-v2":        "Insert the peg into the hole from the side.",
    "window-open-v2":            "Slide the window open.",
    "window-close-v2":           "Slide the window closed.",
    "assembly-v2":               "Insert the wrench into the peg.",
    "basketball-v2":             "Dunk the basketball into the basket.",
    "bin-picking-v2":            "Pick the object from one bin and place it in the other.",
    "box-close-v2":              "Close the lid of the box.",
    "coffee-button-v2":          "Press the coffee button.",
    "coffee-pull-v2":            "Pull the mug to the goal position.",
    "coffee-push-v2":            "Push the mug to the goal position.",
    "dial-turn-v2":              "Turn the dial to the goal angle.",
    "disassemble-v2":            "Remove the wrench from the peg.",
    "door-close-v2":             "Close the door.",
    "faucet-open-v2":            "Turn the faucet to the left to open it.",
    "faucet-close-v2":           "Turn the faucet to the right to close it.",
    "hammer-v2":                 "Drive the nail with the hammer.",
    "hand-insert-v2":            "Insert the peg into the hole.",
    "lever-pull-v2":             "Pull the lever downward.",
    "plate-slide-v2":            "Slide the plate to the goal position.",
    "soccer-v2":                 "Kick the soccer ball into the goal.",
    "stick-push-v2":             "Push the puck with the stick to the goal.",
    "sweep-v2":                  "Sweep the puck to the goal.",
    "sweep-into-v2":             "Sweep the puck into the goal region.",
}


class MetaWorldConverter(BaseConverter):
    """
    Converts MetaWorld demonstration data → GST-VLA canonical format.

    Supports:
      - Pre-recorded HDF5 files (fast)
      - Live rollout collection via MetaWorld env (slow but flexible)
    """

    def __init__(
        self,
        source_root:  str,
        output_root:  str,
        tasks:        Optional[List[str]] = None,
        live_collect: bool  = False,
        n_demos_live: int   = 50,
        use_depth:    bool  = False,   # MetaWorld depth requires custom render
        max_episodes: Optional[int] = None,
        overwrite:    bool  = False,
        img_size:     int   = 224,
        verbose:      bool  = True,
    ):
        super().__init__(
            source_root=source_root,
            output_root=output_root,
            source_name="metaworld",
            max_episodes=max_episodes,
            overwrite=overwrite,
            img_size=img_size,
            verbose=verbose,
        )
        self.tasks        = tasks or MT10_TASKS
        self.live_collect = live_collect
        self.n_demos_live = n_demos_live
        self.use_depth    = use_depth

    # ─────────────────────────────────────────
    # Episode discovery
    # ─────────────────────────────────────────

    def get_episode_list(self) -> List[Tuple[str, Path, int]]:
        """Returns list of (task_name, hdf5_path, demo_idx)."""
        if self.live_collect:
            return self._get_live_episode_list()

        episodes = []
        source   = Path(self.source_root)

        for task_name in self.tasks:
            # Try task-named subdirectory
            task_dirs = list(source.glob(f"**/{task_name}")) + \
                        list(source.glob(f"**/{task_name.replace('-v2','')}"))
            if not task_dirs:
                task_dirs = [source]

            for task_dir in task_dirs:
                hdf5_files = sorted(task_dir.glob("*.hdf5")) + \
                             sorted(task_dir.glob("*.h5"))
                for hdf5_path in hdf5_files:
                    n = self._count_hdf5_demos(hdf5_path)
                    for i in range(n):
                        episodes.append((task_name, hdf5_path, i))
                    logger.info(f"  [{task_name}] {hdf5_path.name}: {n} demos")

        if not episodes:
            logger.warning(f"No HDF5 files found in {self.source_root}. "
                           "Try --live or check your path.")
        logger.info(f"[MetaWorld] Total episodes: {len(episodes)}")
        return episodes

    def _count_hdf5_demos(self, path: Path) -> int:
        try:
            import h5py
            with h5py.File(str(path), "r") as f:
                return len(f.get("data", {}).keys())
        except Exception:
            return 0

    def _get_live_episode_list(self) -> List[Tuple[str, None, int]]:
        """For live collection, return (task_name, None, demo_idx) placeholders."""
        episodes = []
        for task_name in self.tasks:
            for i in range(self.n_demos_live):
                episodes.append((task_name, None, i))
        return episodes

    # ─────────────────────────────────────────
    # Raw episode loading
    # ─────────────────────────────────────────

    def load_raw_episode(
        self, episode_id: Tuple[str, Optional[Path], int]
    ) -> Dict[str, Any]:
        """Load one MetaWorld episode (HDF5 or live)."""
        task_name, hdf5_path, demo_idx = episode_id

        if hdf5_path is None:
            # Live collection
            return self._collect_live_episode(task_name, demo_idx)

        # HDF5 loading
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required: pip install h5py")

        with h5py.File(str(hdf5_path), "r") as f:
            data = f.get("data", f)
            traj_keys = sorted(data.keys(),
                               key=lambda k: int(k.split("_")[-1]))
            if demo_idx >= len(traj_keys):
                return {}
            traj = data[traj_keys[demo_idx]]
            obs  = traj.get("obs", traj)

            raw = {"task_name": task_name}

            # RGB
            for img_key in ["image", "rgb", "pixels", "agentview_rgb"]:
                if img_key in obs:
                    raw["rgb"] = obs[img_key][:]
                    break
            else:
                raw["rgb"] = None

            # State
            for state_key in ["state", "robot_state", "proprio"]:
                if state_key in obs:
                    raw["state"] = obs[state_key][:]
                    break
            else:
                raw["state"] = np.zeros((10, 39))

            # Actions — MetaWorld actions are typically (T, 4): [Δx,Δy,Δz, grip]
            raw["actions"]   = traj["actions"][:] if "actions" in traj else None
            raw["rewards"]   = traj["rewards"][:] if "rewards" in traj else None
            raw["terminals"] = traj["terminals"][:] if "terminals" in traj else None

            # Language
            if "language_goal" in traj:
                lang = traj["language_goal"]
                raw["instruction"] = str(lang[0] if hasattr(lang, "__len__") else lang)
            else:
                raw["instruction"] = METAWORLD_INSTRUCTIONS.get(
                    task_name, task_name.replace("-", " ").capitalize() + "."
                )

        return raw

    def _collect_live_episode(
        self, task_name: str, demo_idx: int
    ) -> Dict[str, Any]:
        """
        Collect one episode via MetaWorld scripted policy.
        Requires `metaworld` package installed.
        """
        try:
            import metaworld
            import metaworld.policies as policies
        except ImportError:
            raise ImportError(
                "metaworld package required for live collection.\n"
                "Install: pip install git+https://github.com/Farama-Foundation/Metaworld"
            )

        # Build env
        task_cls_name = task_name.replace("-", "_").replace("v2", "v2")
        task_cls_name = "".join(w.capitalize() for w in task_cls_name.split("_"))
        if not task_cls_name.endswith("V2"):
            task_cls_name += "V2"

        env_cls = getattr(metaworld.envs, task_cls_name, None)
        if env_cls is None:
            logger.warning(f"[MetaWorld] Unknown task class: {task_cls_name}")
            return {}

        # Find scripted policy
        policy_name = f"Sawyer{task_cls_name}Policy"
        policy_cls  = getattr(policies, policy_name, None)
        if policy_cls is None:
            logger.warning(f"[MetaWorld] No scripted policy for {task_name}")
            return {}

        env    = env_cls()
        policy = policy_cls()

        np.random.seed(demo_idx)
        env.set_task(env.train_tasks[demo_idx % len(env.train_tasks)])
        obs  = env.reset()

        rgb_frames, state_frames, action_frames = [], [], []
        reward_total = 0.0
        T_max = 500

        for _ in range(T_max):
            # Render RGB at 84×84
            frame = env.render(offscreen=True, resolution=(84, 84))
            if frame is not None:
                rgb_frames.append(frame)

            # Get state
            state_frames.append(obs.copy())

            # Scripted action
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            action_frames.append(action.copy())
            reward_total += reward

            if done:
                break

        env.close()

        if not rgb_frames:
            return {}

        T = len(rgb_frames)
        return {
            "task_name":   task_name,
            "rgb":         np.stack(rgb_frames).transpose(0, 3, 1, 2),  # (T, 3, H, W)
            "state":       np.stack(state_frames[:T]),
            "actions":     np.stack(action_frames[:T]),
            "rewards":     np.array([0.0] * (T - 1) + [reward_total]),
            "terminals":   np.array([False] * (T - 1) + [True]),
            "instruction": METAWORLD_INSTRUCTIONS.get(task_name, task_name),
        }

    # ─────────────────────────────────────────
    # Episode conversion
    # ─────────────────────────────────────────

    def convert_episode(
        self,
        raw: Dict[str, Any],
        episode_id: Any,
    ) -> Optional[EpisodeData]:
        """Convert raw MetaWorld episode to canonical EpisodeData."""
        if not raw or raw.get("rgb") is None:
            return None

        rgb_raw = raw["rgb"]
        if rgb_raw.ndim == 4 and rgb_raw.shape[1] == 3:
            rgb_raw = rgb_raw.transpose(0, 2, 3, 1)   # (T, H, W, 3)
        T = len(rgb_raw)
        if T < 2:
            return None

        # RGB
        rgb_frames = [resize_rgb(rgb_raw[t], self.img_size) for t in range(T)]

        # No depth for MetaWorld by default
        depth_frames = None

        # Robot state from MetaWorld proprioceptive obs (T, 39)
        # First 18 dims: robot [qpos(9), qvel(9)]
        state_raw = raw.get("state", np.zeros((T, 39))).astype(np.float32)[:T]
        qpos = state_raw[:, :7]   # joint positions (7)
        qvel = state_raw[:, 9:16] if state_raw.shape[1] > 16 else np.zeros((T, 7))
        robot_states = np.concatenate([qpos, qvel], axis=-1)  # (T, 14)

        # Actions: MetaWorld uses (T, 4) [Δx, Δy, Δz, grip]
        # Extend to 7-dim by padding rotation deltas with zeros
        if raw.get("actions") is not None:
            acts_raw = raw["actions"].astype(np.float32)[:T]  # (T, 4)
            # Pad to 7: [Δx,Δy,Δz, Δroll=0,Δpitch=0,Δyaw=0, grip]
            delta_xyz  = acts_raw[:, :3]
            grip       = acts_raw[:, 3:4]
            delta_rot  = np.zeros((T, 3), dtype=np.float32)
            actions    = np.concatenate([delta_xyz, delta_rot, grip], axis=-1)  # (T, 7)
        else:
            actions = np.zeros((T, 7), dtype=np.float32)

        # Camera K
        camera_K = build_K_from_fov(self.img_size, self.img_size, METAWORLD_CAMERA_FOV)

        # Success
        success = False
        if raw.get("rewards") is not None:
            success = float(raw["rewards"][-1]) > 0.5 or float(raw["rewards"].max()) > 0.5

        task_name = raw["task_name"]
        meta = EpisodeMeta(
            episode_id     = "",
            task_name      = task_name,
            instruction    = raw["instruction"],
            source_dataset = "metaworld",
            source_path    = str(episode_id),
            n_frames       = T,
            success        = success,
            robot_type     = "sawyer",
            camera_name    = "front",
            has_depth      = False,
        )

        return EpisodeData(
            meta=meta,
            rgb_frames=rgb_frames,
            depth_frames=depth_frames,
            actions=actions,
            robot_states=robot_states,
            camera_K=camera_K,
        )
