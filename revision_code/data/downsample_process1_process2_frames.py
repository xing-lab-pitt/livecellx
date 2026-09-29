#!/usr/bin/env python
"""Temporally downsample paired Process1/Process2 images and masks.

The source data are read from each dataset's ``2D_8bit`` and
``2D_mask_renamed`` directories. With the default settings, original frames
1, 6, 11, 16, ... are retained (stride 5). The retained pairs are copied and
renumbered as a normal consecutive sequence starting at 1:

    original 1  -> Fused_0001.tiff / Fused_0001_cp_masks.png
    original 6  -> Fused_0002.tiff / Fused_0002_cp_masks.png
    original 11 -> Fused_0003.tiff / Fused_0003_cp_masks.png

Outputs are written to new dataset roots by default, so the source data are
never modified:

    process1_30den_downsample5/{2D_8bit,2D_mask_renamed}
    process2_downsample5/{2D_8bit,2D_mask_renamed}

Each output root also receives ``frame_mapping.csv`` recording the original
and renumbered frame indices. Existing non-empty output roots are rejected
unless ``--overwrite`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
FRAME_PATTERN = re.compile(r"^Fused_(\d+)(?:_cp_masks)?\.(?:tif|tiff|png)$", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_root: Path


DATASETS = {
    "process1": DatasetSpec("process1", SCRIPT_DIR / "process1_30den"),
    "process2": DatasetSpec("process2", SCRIPT_DIR / "process2"),
}


def indexed_files(directory: Path, expected_mask: bool) -> Dict[int, Path]:
    """Index valid image or mask files by the frame number in the filename."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    indexed: Dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = FRAME_PATTERN.match(path.name)
        if match is None:
            continue
        is_mask = "_cp_masks" in path.stem
        if is_mask != expected_mask:
            continue
        frame = int(match.group(1))
        if frame in indexed:
            raise ValueError(
                f"Multiple {'mask' if expected_mask else 'image'} files for frame "
                f"{frame}: {indexed[frame]} and {path}"
            )
        indexed[frame] = path
    if not indexed:
        raise FileNotFoundError(f"No matching files found in {directory}")
    return indexed


def selected_frames(common_frames: Iterable[int], start_frame: int, stride: int) -> List[int]:
    """Select start_frame, start_frame + stride, ... from available frame IDs."""
    frame_set = set(common_frames)
    if start_frame not in frame_set:
        raise ValueError(f"Requested starting frame {start_frame} is not available")
    last_frame = max(frame_set)
    requested = list(range(start_frame, last_frame + 1, stride))
    missing = [frame for frame in requested if frame not in frame_set]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(
            f"The requested temporal sequence contains missing image-mask pairs: {preview}"
        )
    return requested


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_root}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    (output_root / "2D_8bit").mkdir(parents=True, exist_ok=True)
    (output_root / "2D_mask_renamed").mkdir(parents=True, exist_ok=True)


def downsample_dataset(
    spec: DatasetSpec,
    output_root: Path,
    start_frame: int,
    stride: int,
    overwrite: bool,
) -> None:
    image_dir = spec.source_root / "2D_8bit"
    mask_dir = spec.source_root / "2D_mask_renamed"
    images = indexed_files(image_dir, expected_mask=False)
    masks = indexed_files(mask_dir, expected_mask=True)

    image_only = sorted(set(images) - set(masks))
    mask_only = sorted(set(masks) - set(images))
    if image_only or mask_only:
        raise ValueError(
            f"{spec.name} has unpaired files. Image-only frames: {image_only[:10]}; "
            f"mask-only frames: {mask_only[:10]}"
        )

    keep = selected_frames(images.keys(), start_frame=start_frame, stride=stride)
    prepare_output_root(output_root, overwrite=overwrite)
    output_images = output_root / "2D_8bit"
    output_masks = output_root / "2D_mask_renamed"

    rows = []
    for new_frame, original_frame in enumerate(keep, start=1):
        source_image = images[original_frame]
        source_mask = masks[original_frame]
        image_suffix = source_image.suffix.lower()
        output_image = output_images / f"Fused_{new_frame:04d}{image_suffix}"
        output_mask = output_masks / f"Fused_{new_frame:04d}_cp_masks.png"
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
        f"[{spec.name}] copied {len(rows)} paired frames from "
        f"{min(keep)}..{max(keep)} to {output_root}"
    )
    print(f"[{spec.name}] mapping: {mapping_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help="Datasets to process (default: process1 process2).",
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=SCRIPT_DIR,
        help="Parent directory for the new downsampled dataset roots.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate an existing downsampled output root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be a positive integer")
    if args.start_frame < 0:
        raise ValueError("--start-frame cannot be negative")

    output_base = args.output_base.resolve()
    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        source_suffix = spec.source_root.name
        output_root = output_base / f"{source_suffix}_downsample{args.stride}"
        downsample_dataset(
            spec,
            output_root,
            start_frame=args.start_frame,
            stride=args.stride,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
