"""
Convert RLBench → GST-VLA canonical format
============================================
Usage:
    python data/scripts/convert_rlbench.py \\
        --source /data/rlbench_demos \\
        --output /data/gst_vla/rlbench \\
        --tasks reach_target pick_and_lift open_drawer push_button \\
        --camera front \\
        --img_size 224 \\
        --n_variations 3

    # Dry run to check structure
    python data/scripts/convert_rlbench.py \\
        --source /data/rlbench_demos --output /tmp/test \\
        --dry_run

    # All tasks in a directory
    python data/scripts/convert_rlbench.py \\
        --source /data/rlbench_demos --output /data/gst_vla/rlbench \\
        --all_tasks --max_episodes 1000
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.converters.rlbench_converter import RLBenchConverter, VALID_CAMERAS


def parse_args():
    p = argparse.ArgumentParser(description="Convert RLBench to GST-VLA format")
    p.add_argument("--source",        required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--tasks",         nargs="+", default=None,
                   help="Task names to include (default: all found)")
    p.add_argument("--all_tasks",     action="store_true",
                   help="Include all task directories found in source")
    p.add_argument("--camera",        default="front", choices=VALID_CAMERAS)
    p.add_argument("--img_size",      type=int, default=224)
    p.add_argument("--n_variations",  type=int, default=None,
                   help="Max variations per task (None = all)")
    p.add_argument("--max_episodes",  type=int, default=None)
    p.add_argument("--no_depth",      action="store_true")
    p.add_argument("--overwrite",     action="store_true")
    p.add_argument("--dry_run",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Discover tasks if --all_tasks
    tasks = args.tasks
    if args.all_tasks:
        source = Path(args.source)
        tasks  = sorted([d.name for d in source.iterdir() if d.is_dir()])

    print("=" * 60)
    print("  RLBench → GST-VLA Converter")
    print("=" * 60)
    print(f"  Source:      {args.source}")
    print(f"  Output:      {args.output}")
    print(f"  Tasks:       {tasks or 'all found'}")
    print(f"  Camera:      {args.camera}")
    print(f"  Image size:  {args.img_size}×{args.img_size}")
    print(f"  Variations:  {args.n_variations or 'all'}")
    print(f"  Max eps:     {args.max_episodes or 'unlimited'}")
    print(f"  Use depth:   {not args.no_depth}")
    print("=" * 60)

    converter = RLBenchConverter(
        source_root  = args.source,
        output_root  = args.output,
        tasks        = tasks,
        camera       = args.camera,
        use_depth    = not args.no_depth,
        n_variations = args.n_variations,
        max_episodes = args.max_episodes,
        overwrite    = args.overwrite,
        img_size     = args.img_size,
        verbose      = True,
    )

    if args.dry_run:
        print("\n[DRY RUN] Scanning episodes...")
        episodes = converter.get_episode_list()
        print(f"  Found {len(episodes)} episodes (no files written)")
        task_counts = Counter(ep[0] for ep in episodes)
        print("\n  Task distribution:")
        for task, count in sorted(task_counts.items()):
            print(f"    {task:<45} {count:4d}")
        return

    stats = converter.run()
    print(f"\n  ✓ Conversion complete → {args.output}")
    print(f"    Depth range: [{stats.depth_p5:.2f}, {stats.depth_p95:.2f}] m")


if __name__ == "__main__":
    main()
