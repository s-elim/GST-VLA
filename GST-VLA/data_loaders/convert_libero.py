"""
Convert LIBERO → GST-VLA canonical format
==========================================
Usage:
    python data/scripts/convert_libero.py \\
        --source /data/libero \\
        --output /data/gst_vla/libero \\
        --suites libero_spatial libero_object libero_goal libero_long \\
        --camera agentview \\
        --img_size 224 \\
        --max_episodes 500

    # Quick test (5 episodes)
    python data/scripts/convert_libero.py \\
        --source /data/libero --output /tmp/libero_test \\
        --max_episodes 5 --dry_run
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loaders.libero_converter import LIBEROConverter


def parse_args():
    p = argparse.ArgumentParser(description="Convert LIBERO to GST-VLA format")
    p.add_argument("--source",        required=True,
                   help="Root dir containing libero_spatial/, libero_object/, etc.")
    p.add_argument("--output",        required=True,
                   help="Output directory for canonical format")
    p.add_argument("--suites",        nargs="+", default=None,
                   help="LIBERO suites to include (default: all found)")
    p.add_argument("--camera",        default="agentview",
                   choices=["agentview", "robot0_eye_in_hand"],
                   help="Camera to use for RGB/depth")
    p.add_argument("--img_size",      type=int, default=224)
    p.add_argument("--max_episodes",  type=int, default=None)
    p.add_argument("--no_depth",      action="store_true")
    p.add_argument("--action_type",   default="delta", choices=["delta", "raw"])
    p.add_argument("--overwrite",     action="store_true")
    p.add_argument("--dry_run",       action="store_true",
                   help="Scan episodes without writing anything")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  LIBERO → GST-VLA Converter")
    print("=" * 60)
    print(f"  Source:      {args.source}")
    print(f"  Output:      {args.output}")
    print(f"  Suites:      {args.suites or 'all'}")
    print(f"  Camera:      {args.camera}")
    print(f"  Image size:  {args.img_size}×{args.img_size}")
    print(f"  Max eps:     {args.max_episodes or 'unlimited'}")
    print(f"  Use depth:   {not args.no_depth}")
    print(f"  Action type: {args.action_type}")
    print(f"  Dry run:     {args.dry_run}")
    print("=" * 60)

    converter = LIBEROConverter(
        source_root   = args.source,
        output_root   = args.output,
        suites        = args.suites,
        camera        = args.camera,
        use_depth     = not args.no_depth,
        action_type   = args.action_type,
        max_episodes  = args.max_episodes,
        overwrite     = args.overwrite,
        img_size      = args.img_size,
        verbose       = True,
    )

    if args.dry_run:
        print("\n[DRY RUN] Scanning episodes...")
        episodes = converter.get_episode_list()
        print(f"  Found {len(episodes)} episodes (no files written)")

        # Show task distribution
        from collections import Counter
        suite_counts = Counter(ep[1] for ep in episodes)
        print("\n  Suite distribution:")
        for suite, count in sorted(suite_counts.items()):
            print(f"    {suite:<30} {count:4d} episodes")
        return

    stats = converter.run()

    print("\n  Normalization stats:")
    print(f"    Action mean:  {[f'{v:.4f}' for v in stats.action_mean]}")
    print(f"    Action std:   {[f'{v:.4f}' for v in stats.action_std]}")
    print(f"    Depth range:  [{stats.depth_p5:.2f}, {stats.depth_p95:.2f}] m")
    print(f"\n  ✓ Conversion complete → {args.output}")


if __name__ == "__main__":
    main()
