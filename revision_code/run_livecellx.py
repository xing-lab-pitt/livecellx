#!/usr/bin/env python
"""
LivecellX pipeline: SORT tracking + CS-Net multi-round correction.
Run in the `livecellx` conda env.

Usage:
    conda run -n livecellx python run_livecellx.py
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm

from livecellx.core.datasets import LiveCellImageDataset
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.core import SingleCellStatic, SingleCellTrajectoryCollection
from livecellx.core.single_cell import get_time2scs
from livecellx.track.sort_tracker_utils import track_SORT_bbox_from_scs
from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux
from livecellx.model_zoo.segmentation.custom_transforms import CustomTransformEdtV9


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="LivecellX tracking + correction pipeline")
    parser.add_argument("--image_dir", type=str,
                        default=str(SCRIPT_DIR / "data/process1_30den/2D"))
    parser.add_argument("--mask_dir", type=str,
                        default=str(SCRIPT_DIR / "data/process1_30den/2D_mask"))
    parser.add_argument("--model_ckpt", type=str,
                        default=str(SCRIPT_DIR / "model/last.ckpt"))
    parser.add_argument("--backbone", type=str, default="deeplabV3",
                        choices=["unet_aux", "deeplabV3"])
    parser.add_argument("--output_dir", type=str,
                        default=str(SCRIPT_DIR / "results"))

    # Tracking
    parser.add_argument("--max_age", type=int, default=5)
    parser.add_argument("--min_hits", type=int, default=1)

    # CS-Net correction
    parser.add_argument("--max_round", type=int, default=50)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--match_search_interval", type=int, default=3)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--h_threshold", type=float, default=0.3)
    parser.add_argument("--area_threshold", type=float, default=1000)

    return parser.parse_args()


# ─── Data loading ────────────────────────────────────────────────────────────

def extract_frame_number(filepath):
    """Extract numeric frame index from filenames like Fused_123.tiff."""
    match = re.search(r'(\d+)', Path(filepath).stem)
    return int(match.group(1)) if match else None


def load_datasets(image_dir, mask_dir):
    """Load images and masks into LiveCellImageDataset objects."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    # Discover image files, exclude Cellpose byproducts
    img_files = []
    for ext in ["*.tif", "*.tiff", "*.png"]:
        for f in image_dir.glob(ext):
            if "_seg" not in f.stem and "_flows" not in f.stem and "_cp_" not in f.stem:
                img_files.append(f)

    if not img_files:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    img_files.sort(key=lambda f: extract_frame_number(f))
    img_time2url = {}
    for f in img_files:
        t = extract_frame_number(f)
        if t is not None:
            img_time2url[t] = str(f)

    # Discover mask files
    mask_files = []
    for ext in ["*.tif", "*.tiff", "*.png"]:
        mask_files.extend(mask_dir.glob(ext))

    mask_time2url = {}
    for f in mask_files:
        t = extract_frame_number(f)
        if t is not None and t in img_time2url:
            mask_time2url[t] = str(f)

    common_times = sorted(set(img_time2url.keys()) & set(mask_time2url.keys()))
    img_time2url = {t: img_time2url[t] for t in common_times}
    mask_time2url = {t: mask_time2url[t] for t in common_times}

    print(f"Found {len(common_times)} matched image-mask pairs "
          f"(time {common_times[0]}-{common_times[-1]})")

    img_dataset = LiveCellImageDataset(time2url=img_time2url, name="raw_images")
    mask_dataset = LiveCellImageDataset(time2url=mask_time2url, name="cellpose_masks")
    return img_dataset, mask_dataset


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_trajectory_metrics(sctc):
    """Return per-trajectory metrics as a list of dicts."""
    per_traj = []
    for tid, sct in sctc:
        times = sct.times
        if len(times) == 0:
            continue
        length = len(times)
        span = times[-1] - times[0] + 1
        missing = span - length
        vacancy = missing / span if span > 0 else 0
        per_traj.append({
            "track_id": tid,
            "length": length,
            "span": span,
            "missing_frames": missing,
            "vacancy_rate": vacancy,
            "start_time": times[0],
            "end_time": times[-1],
        })
    return per_traj


def summarize_metrics(per_traj):
    """Compute aggregate metrics from per-trajectory list."""
    if not per_traj:
        return {}
    lengths = [t["length"] for t in per_traj]
    vacancies = [t["vacancy_rate"] for t in per_traj]
    return {
        "num_trajectories": len(per_traj),
        "num_cells": sum(t["length"] for t in per_traj),
        "mean_traj_length": float(np.mean(lengths)),
        "median_traj_length": float(np.median(lengths)),
        "mean_vacancy_rate": float(np.mean(vacancies)),
        "median_vacancy_rate": float(np.median(vacancies)),
        "track_continuity": float(np.mean([1 if v < 0.1 else 0 for v in vacancies])),
    }


# ─── Export corrected masks ─────────────────────────────────────────────────

def export_corrected_masks(sctc, img_dataset, output_dir):
    """Export corrected segmentation masks as label images per frame."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scs = sctc.get_all_scs()
    time2scs = get_time2scs(all_scs)

    times = sorted(img_dataset.time2url.keys())
    for t in tqdm.tqdm(times, desc="Exporting corrected masks"):
        img = img_dataset.get_img_by_time(t)
        H, W = img.shape[:2]
        label_mask = np.zeros((H, W), dtype=np.uint16)

        if t in time2scs:
            for idx, sc in enumerate(time2scs[t], start=1):
                try:
                    # get_contour_mask(crop=False) returns full-image-size mask
                    mask = sc.get_contour_mask(padding=0, crop=False)
                    label_mask[mask > 0] = idx
                except Exception:
                    pass

        cv2.imwrite(str(output_dir / f"frame_{t:04d}.png"), label_mask)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading images and Cellpose masks")
    print("=" * 60)
    img_dataset, mask_dataset = load_datasets(args.image_dir, args.mask_dir)

    print("Converting masks to SingleCellStatic objects...")
    all_scs = prep_scs_from_mask_dataset(mask_dataset, img_dataset)
    print(f"  Created {len(all_scs)} SingleCellStatic objects")

    # ── SORT tracking ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: SORT tracking")
    print("=" * 60)
    sctc_before = track_SORT_bbox_from_scs(
        all_scs, raw_imgs=img_dataset,
        max_age=args.max_age, min_hits=args.min_hits, sc_inplace=True,
    )
    print(f"  Trajectories: {len(sctc_before)}")
    print(f"  Total cells:  {len(sctc_before.get_all_scs())}")

    # Save before-correction metrics
    before_per_traj = compute_trajectory_metrics(sctc_before)
    before_summary = summarize_metrics(before_per_traj)
    print(f"  Mean vacancy rate: {before_summary['mean_vacancy_rate']:.4f}")

    dataset_json_dir = output_dir / "datasets"
    sctc_before.write_json(
        output_dir / "livecellx_sctc_before.json",
        dataset_json_dir=dataset_json_dir,
    )

    # The trajectory repair used by the original notebook belongs between SORT
    # and CS-Net. Keep the raw SORT file above as the unmodified baseline.
    print("\n" + "=" * 60)
    print("Step 3: Post-SORT trajectory patches")
    print("=" * 60)
    from livecellx_post_sort_patches import apply_post_sort_patches

    sctc_for_csnet, patch_report = apply_post_sort_patches(sctc_before, all_scs)
    sctc_for_csnet.write_json(
        output_dir / "livecellx_sctc_post_sort_patched.json",
        dataset_json_dir=dataset_json_dir,
    )
    with open(output_dir / "livecellx_post_sort_patch_report.json", "w") as handle:
        json.dump(patch_report, handle, indent=2)
    print(
        f"  Patched trajectories: {len(sctc_for_csnet)}; "
        f"cells: {len(sctc_for_csnet.get_all_scs())}"
    )

    # ── CS-Net correction ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: CS-Net correction")
    print("=" * 60)

    # Fix: checkpoint saved with old pandas/torch. Patch loading for compatibility.
    import pickle
    import torch.serialization
    _orig_internal_load = torch.serialization._load

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "pandas.core.indexes.numeric":
                return pd.Index
            return super().find_class(module, name)

    class _PatchedPickle:
        Unpickler = _CompatUnpickler
        load = pickle.load
        def __getattr__(self, name):
            return getattr(pickle, name)

    def _compat_load(zip_file, map_location, pickle_module, pickle_file="data.pkl", **kw):
        return _orig_internal_load(zip_file, map_location, _PatchedPickle(), pickle_file, **kw)

    torch.serialization._load = _compat_load
    model = CorrectSegNetAux.load_from_checkpoint(args.model_ckpt)
    torch.serialization._load = _orig_internal_load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using CS-Net device: {device}")
    model.to(device)
    model.eval()
    input_transforms = CustomTransformEdtV9(
        degrees=0, shear=0, flip_p=0, use_gaussian_blur=True, gaussian_blur_sigma=15
    )

    # Import the correction function from the benchmark script
    sys.path.insert(0, str(SCRIPT_DIR.parent / "notebooks"))
    from CXA_2D_multiround_correction_with_tracking_benchmark import run_multiround_correction

    class CorrectionArgs:
        pass

    corr_args = CorrectionArgs()
    corr_args.match_threshold = args.match_threshold
    corr_args.match_search_interval = args.match_search_interval
    corr_args.max_round = args.max_round
    corr_args.area_threshold = args.area_threshold
    corr_args.h_threshold = args.h_threshold
    corr_args.out_threshold = 1.0
    corr_args.padding = args.padding
    corr_args.save_masks = False
    corr_args.enable_viz = False
    corr_args.minimal_output = True
    corr_args.dist_to_boundary = 50

    sctc_after, round_df_dict, _ = run_multiround_correction(
        sctc_for_csnet, model, input_transforms, output_dir, corr_args
    )

    sctc_after.write_json(
        output_dir / "livecellx_sctc_after.json",
        dataset_json_dir=dataset_json_dir,
    )
    pd.DataFrame(round_df_dict).to_csv(output_dir / "livecellx_correction_rounds.csv", index=False)

    # After-correction metrics
    after_per_traj = compute_trajectory_metrics(sctc_after)
    after_summary = summarize_metrics(after_per_traj)
    print(f"  After correction — mean vacancy rate: {after_summary['mean_vacancy_rate']:.4f}")

    # ── Export corrected masks for visual comparison ─────────────────────────
    print("\nExporting corrected masks...")
    export_corrected_masks(sctc_after, img_dataset, output_dir / "livecellx_corrected_masks")

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {
        "before_correction": {
            "summary": before_summary,
            "per_trajectory": before_per_traj,
        },
        "after_correction": {
            "summary": after_summary,
            "per_trajectory": after_per_traj,
        },
        "config": {
            "max_age": args.max_age,
            "min_hits": args.min_hits,
            "post_sort_patches_applied": True,
            "backbone": args.backbone,
            "max_round": args.max_round,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    metrics["post_sort_patches"] = patch_report
    with open(output_dir / "livecellx_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    # Clean up
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
