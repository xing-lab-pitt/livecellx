#!/usr/bin/env python
"""Downsample the Process2 alternative1 image/mask pair every five frames.

This script follows the same selection and renaming rules as
downsample_process1_process2_frames.py:

    original frames 1, 6, 11, 16, ...
        -> downsampled frames 1, 2, 3, 4, ...

Inputs:
    revision_code/data/process2/2D_8bit
    revision_code/data/process2/2D_mask_alternative1_renamed

Default output:
    revision_code/data/process2_alternative1_downsample5/
        2D_8bit/
        2D_mask_renamed/
        frame_mapping.csv

The source data are never modified. A non-empty output folder is rejected
unless --overwrite is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from downsample_process1_process2_frames import (
    indexed_files,
    prepare_output_root,
    selected_frames,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR / "process2"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "process2_alternative1_downsample5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate a non-empty output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be a positive integer")
    if args.start_frame < 0:
        raise ValueError("--start-frame cannot be negative")

    image_dir = SOURCE_ROOT / "2D_8bit"
    mask_dir = SOURCE_ROOT / "2D_mask_alternative1_renamed"
    output_root = args.output_root.resolve()

    images = indexed_files(image_dir, expected_mask=False)
    masks = indexed_files(mask_dir, expected_mask=True)

    image_only = sorted(set(images) - set(masks))
    mask_only = sorted(set(masks) - set(images))
    if image_only or mask_only:
        raise ValueError(
            "Process2 alternative1 has unpaired files. "
            f"Image-only frames: {image_only[:10]}; "
            f"mask-only frames: {mask_only[:10]}"
        )

    keep = selected_frames(
        images.keys(),
        start_frame=int(args.start_frame),
        stride=int(args.stride),
    )
    prepare_output_root(output_root, overwrite=bool(args.overwrite))
    output_images = output_root / "2D_8bit"
    output_masks = output_root / "2D_mask_renamed"

    rows = []
    for new_frame, original_frame in enumerate(keep, start=1):
        source_image = images[original_frame]
        source_mask = masks[original_frame]
        output_image = (
            output_images
            / f"Fused_{new_frame:04d}{source_image.suffix.lower()}"
        )
        output_mask = (
            output_masks / f"Fused_{new_frame:04d}_cp_masks.png"
        )
        shutil.copy2(source_image, output_image)
        shutil.copy2(source_mask, output_mask)
        rows.append(
            {
                "downsampled_frame": new_frame,
                "original_frame": original_frame,
                "source_image": str(source_image.resolve()),
                "source_mask": str(source_mask.resolve()),
                "output_image": str(output_image.resolve()),
                "output_mask": str(output_mask.resolve()),
            }
        )

    mapping_path = output_root / "frame_mapping.csv"
    with mapping_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[process2_alternative1] copied {len(rows)} paired frames "
        f"({min(keep)}, {min(keep) + args.stride}, ... {max(keep)}) "
        f"to {output_root}"
    )
    print(f"[mapping] {mapping_path}")


if __name__ == "__main__":
    main()
