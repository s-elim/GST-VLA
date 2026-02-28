"""
Convert MetaWorld → GST-VLA canonical format
==============================================
Usage:
    # From pre-recorded HDF5 files
    python data/scripts/convert_metaworld.py \\
        --source /data/metaworld_demos \\
        --output /data/gst_vla/metaworld \\
        --tasks reach-v2 push-v2 pick-place-v2

    # Live collection (requires metaworld package)
    python data/scripts/convert_metaworld.py \\
        --live \\
        --tasks MT10 \\
        --n_demos 50 \\
        --output /data/gst_vla/metaworld

    # MT10 with all tasks
    python data/scripts/convert_metaworld.py \\
        --source /data/metaworld --output /data/gst_vla/metaworld \\
        --tasks MT10 --max_episodes 500
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.converters.metaworld_converter import (
    MetaWorldConverter, MT10_TASKS
)


def parse_args():
    p = argparse.ArgumentParser(description="Convert MetaWorld to GST-VLA format")
    p.add_argument("--source",        default="",
                   help="Root dir with MetaWorld HDF5 files (not needed with --live)")
    p.add_argument("--output",        required=True)
    p.add_argument("--tasks",         nargs="+", default=None,
                   help="Task names or 'MT10' for standard 10 tasks")
    p.add_argument("--live",          action="store_true",
                   help="Collect live via scripted MetaWorld policies")
    p.add_argument("--n_demos",       type=int, default=50,
                   help="Demos per task in live mode")
    p.add_argument("--img_size",      type=int, default=224)
    p.add_argument("--max_episodes",  type=int, default=None)
    p.add_argument("--overwrite",     action="store_true")
    p.add_argument("--dry_run",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve task list
    tasks = args.tasks
    if tasks and len(tasks) == 1 and tasks[0] == "MT10":
        tasks = MT10_TASKS

    print("=" * 60)
    print("  MetaWorld → GST-VLA Converter")
    print("=" * 60)
    print(f"  Mode:        {'Live collection' if args.live else 'HDF5'}")
    print(f"  Source:      {args.source or '(live)'}")
    print(f"  Output:      {args.output}")
    print(f"  Tasks:       {tasks or 'MT10 default'}")
    print(f"  Image size:  {args.img_size}×{args.img_size}")
    print(f"  Max eps:     {args.max_episodes or 'unlimited'}")
    print("=" * 60)

    converter = MetaWorldConverter(
        source_root  = args.source or "/tmp/metaworld_placeholder",
        output_root  = args.output,
        tasks        = tasks,
        live_collect = args.live,
        n_demos_live = args.n_demos,
        max_episodes = args.max_episodes,
        overwrite    = args.overwrite,
        img_size     = args.img_size,
        verbose      = True,
    )

    if args.dry_run:
        print("\n[DRY RUN] Scanning episodes...")
        episodes = converter.get_episode_list()
        print(f"  Would process {len(episodes)} episodes (no files written)")
        from collections import Counter
        task_counts = Counter(ep[0] for ep in episodes)
        print("\n  Task distribution:")
        for task, count in sorted(task_counts.items()):
            print(f"    {task:<40} {count:4d}")
        return

    stats = converter.run()
    print(f"\n  ✓ Conversion complete → {args.output}")


if __name__ == "__main__":
    main()
