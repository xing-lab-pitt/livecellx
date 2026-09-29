#!/usr/bin/env python
"""Run SORT tracking for any paired image/mask dataset.

Example:
    python -u revision_code/run_sort_tracking.py \
        --image-dir /path/to/2D_8bit \
        --mask-dir /path/to/2D_mask_renamed \
        --output-dir /path/to/results

The loadable trajectory collection is saved as
``OUTPUT_DIR/traj_collections/sort.json``. This script performs SORT only; it
does not run trajectory repair, CS-Net, or TimeSformer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.track.sort_tracker_utils import track_SORT_bbox_from_scs

from run_livecellx import load_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-age", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    mask_dir = args.mask_dir.resolve()
    output_dir = args.output_dir.resolve()
    trajectory_dir = output_dir / "traj_collections"
    datasets_dir = trajectory_dir / "datasets"
    output_path = trajectory_dir / "sort.json"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")
    if args.max_age < 0:
        raise ValueError("--max-age cannot be negative")
    if args.min_hits <= 0:
        raise ValueError("--min-hits must be positive")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"SORT output already exists: {output_path}; use --force to replace it")

    trajectory_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    print("[SORT] loading paired images and masks", flush=True)
    image_dataset, mask_dataset = load_datasets(image_dir, mask_dir)
    single_cells = prep_scs_from_mask_dataset(mask_dataset, image_dataset)
    print(f"[SORT] segmented cells: {len(single_cells)}", flush=True)

    print(
        f"[SORT] tracking with max_age={args.max_age}, min_hits={args.min_hits}",
        flush=True,
    )
    sctc = track_SORT_bbox_from_scs(
        single_cells,
        raw_imgs=image_dataset,
        mask_dataset=mask_dataset,
        max_age=int(args.max_age),
        min_hits=int(args.min_hits),
        sc_inplace=True,
    )
    sctc.write_json(str(output_path), dataset_json_dir=datasets_dir.resolve())

    manifest = {
        "stage": "SORT_only",
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "output": str(output_path),
        "max_age": int(args.max_age),
        "min_hits": int(args.min_hits),
        "segmented_cells": int(len(single_cells)),
        "trajectory_count": int(len(sctc)),
        "tracked_cells": int(len(sctc.get_all_scs())),
    }
    with (trajectory_dir / "sort_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[SORT] saved: {output_path}", flush=True)
    print(
        f"[SORT] trajectories={manifest['trajectory_count']}, "
        f"tracked_cells={manifest['tracked_cells']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
