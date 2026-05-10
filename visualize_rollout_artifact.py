from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluation.rollout_visualize import plot_rollout_artifact_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize multi-step rollout predictions from one or more rollout evaluation artifacts."
    )
    parser.add_argument(
        "--artifacts",
        nargs="+",
        required=True,
        help="One or more .pt artifacts produced by the rollout evaluator.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where comparison figures will be written.",
    )
    parser.add_argument(
        "--sequence-index",
        type=int,
        default=0,
        help="Sequence/entity index to visualize. Use -1 to automatically pick the hardest sequence by mean JSD.",
    )
    parser.add_argument(
        "--snapshot-index",
        type=int,
        default=-1,
        help="Explicit rollout-horizon index to visualize. Leave at -1 to use snapshot-policy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="For unordered supports, compare up to this many dominant components.",
    )
    parser.add_argument(
        "--sequence-policy",
        choices=("hardest", "lowest_mean_support", "highest_mean_support"),
        default="hardest",
        help="How to choose a sequence when --sequence-index is negative.",
    )
    parser.add_argument(
        "--snapshot-policy",
        choices=("hardest_jsd", "lowest_mean_support", "highest_mean_support"),
        default="hardest_jsd",
        help="How to choose the displayed horizon within the chosen sequence.",
    )
    parser.add_argument(
        "--ordered-support-mode",
        choices=("full", "focus"),
        default="full",
        help="For ordered supports, either show the full support or a focused window around the selected case.",
    )
    parser.add_argument(
        "--ordered-support-pad",
        type=int,
        default=24,
        help="Extra support bins to include on each side when ordered-support-mode=focus.",
    )
    parser.add_argument(
        "--ordered-support-min-window",
        type=int,
        default=64,
        help="Minimum width of the focused support window for ordered supports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = [torch.load(Path(path).expanduser().resolve(), map_location="cpu") for path in args.artifacts]
    output_dir = Path(args.output_dir).expanduser().resolve()
    produced = plot_rollout_artifact_comparison(
        artifacts=artifacts,
        output_dir=output_dir,
        sequence_index=args.sequence_index,
        snapshot_index=args.snapshot_index,
        top_k=args.top_k,
        sequence_policy=args.sequence_policy,
        snapshot_policy=args.snapshot_policy,
        ordered_support_mode=args.ordered_support_mode,
        ordered_support_pad=args.ordered_support_pad,
        ordered_support_min_window=args.ordered_support_min_window,
    )
    print(f"[rollout_viz] wrote {len(produced)} figure(s) to {output_dir}")
    for path in produced:
        print(f"[rollout_viz] {path}")


if __name__ == "__main__":
    main()
