#!/usr/bin/env python
"""Run LiveCellX correction and lineage reconstruction from an existing SORT SCTC.

This is a dataset-independent pipeline. It requires image and mask directories,
an existing SORT trajectory collection, and an output directory. The fixed
order follows ``step5_track_and_napari_correction copy.ipynb``:

    existing SORT collection on the original masks
      -> restore segmented cells omitted by initial SORT
      -> reconnect initial short trajectory fragments by mask IoU
      -> CS-Net mask correction
      -> export corrected label masks
      -> re-run SORT on the corrected masks
      -> restore segmented cells omitted by final SORT
      -> reconnect final short trajectory fragments by mask IoU
      -> repair conservative two-track identity switches
      -> TimeSformer mitosis detection and mother-daughter reconstruction

The initial SORT collection is supplied by ``run_sort_tracking.py`` (or by
``--sort-sctc``). This script then performs the required second SORT pass on
the exported CS-Net-corrected masks.

Outputs:
    OUTPUT_DIR/traj_collections/sort_post_patched.json
    OUTPUT_DIR/traj_collections/livecellx_csnet.json
    OUTPUT_DIR/livecellx_corrected_masks/*.png
    OUTPUT_DIR/traj_collections/livecellx_corrected_sort.json
    OUTPUT_DIR/traj_collections/livecellx_corrected_sort_identity_repaired.json
    OUTPUT_DIR/traj_collections/livecellx_corrected_sort_post_patched.json
    OUTPUT_DIR/traj_collections/livecellx_csnet_timesformer_lineage.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from livecellx.core import SingleCellTrajectoryCollection
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.model_zoo.segmentation.custom_transforms import CustomTransformEdtV9
from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux
from livecellx.track.sort_tracker_utils import track_SORT_bbox_from_scs

from livecellx_post_sort_patches import apply_post_sort_patches, repair_identity_switches
from run_livecellx import export_corrected_masks, load_datasets


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "notebooks"))
CSNET_MODEL = SCRIPT_DIR / "model" / "last.ckpt"
TIMESFORMER_MODEL = SCRIPT_DIR / "model" / "best_acc_top1_epoch_65.pth"
TIMESFORMER_CONFIG = (
    SCRIPT_DIR / "configs" / "timesformer_divst_v15.py"
)


def log(message: str) -> None:
    print(message, flush=True)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def output_is_stale(output: Path, source: Path, force: bool) -> bool:
    return force or not output.exists() or output.stat().st_mtime < source.stat().st_mtime


def load_csnet(checkpoint: Path, device: torch.device):
    """Load the local Lightning checkpoint with old-pandas compatibility."""
    original_internal_load = torch.serialization._load

    class CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "pandas.core.indexes.numeric":
                return pd.Index
            return super().find_class(module, name)

    class PatchedPickle:
        Unpickler = CompatUnpickler
        load = pickle.load

        def __getattr__(self, name):
            return getattr(pickle, name)

    def compat_load(zip_file, map_location, pickle_module, pickle_file="data.pkl", **kwargs):
        return original_internal_load(
            zip_file,
            map_location,
            PatchedPickle(),
            pickle_file,
            **kwargs,
        )

    torch.serialization._load = compat_load
    try:
        model = CorrectSegNetAux.load_from_checkpoint(str(checkpoint))
    finally:
        torch.serialization._load = original_internal_load
    model.to(device)
    model.eval()
    return model


def save_sctc_atomic(
    sctc: SingleCellTrajectoryCollection,
    path: Path,
    datasets_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    sctc.write_json(str(temporary), dataset_json_dir=datasets_dir.resolve())
    os.replace(temporary, path)


def run_post_sort_patches(
    sort_path: Path,
    patched_path: Path,
    datasets_dir: Path,
    all_segmented_cells,
    force: bool,
    report_name: str,
) -> Path:
    if not output_is_stale(patched_path, sort_path, force):
        log(f"[patches] reusing {patched_path}")
        return patched_path

    log(f"[patches] loading SORT collection: {sort_path}")
    sctc = SingleCellTrajectoryCollection.load_from_json_file(str(sort_path), parallel=False)
    sctc, report = apply_post_sort_patches(sctc, all_segmented_cells)
    save_sctc_atomic(sctc, patched_path, datasets_dir)
    with (patched_path.parent / report_name).open("w") as handle:
        json.dump(report, handle, indent=2)
    log(f"[patches] saved: {patched_path}")
    del sctc
    gc.collect()
    return patched_path


def run_identity_switch_repair(
    source_path: Path,
    repaired_path: Path,
    datasets_dir: Path,
    force: bool,
) -> Path:
    if not output_is_stale(repaired_path, source_path, force):
        log(f"[identity repair] reusing {repaired_path}")
        return repaired_path

    log(f"[identity repair] loading corrected SORT collection: {source_path}")
    sctc = SingleCellTrajectoryCollection.load_from_json_file(str(source_path), parallel=False)
    report = repair_identity_switches(sctc)
    save_sctc_atomic(sctc, repaired_path, datasets_dir)
    with (repaired_path.parent / "identity_switch_repair_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    log(
        "[identity repair] repaired "
        f"{report['identity_switches_repaired']} switches: {repaired_path}"
    )
    del sctc
    gc.collect()
    return repaired_path


def run_csnet_stage(
    source_sort_path: Path,
    csnet_path: Path,
    datasets_dir: Path,
    image_dataset,
    args: argparse.Namespace,
) -> Path:
    if not output_is_stale(
        csnet_path,
        source_sort_path,
        args.force_csnet or args.force_patches,
    ):
        log(f"[CS-Net] reusing {csnet_path}")
        return csnet_path

    from CXA_2D_multiround_correction_with_tracking_benchmark import run_multiround_correction

    log(f"[CS-Net] loading initial SORT collection: {source_sort_path}")
    sctc = SingleCellTrajectoryCollection.load_from_json_file(
        str(source_sort_path),
        parallel=False,
    )
    device = torch.device(args.csnet_device if torch.cuda.is_available() else "cpu")
    log(f"[CS-Net] loading model on {device}: {args.csnet_model}")
    model = load_csnet(Path(args.csnet_model).resolve(), device)
    transforms = CustomTransformEdtV9(
        degrees=0,
        shear=0,
        flip_p=0,
        use_gaussian_blur=True,
        gaussian_blur_sigma=15,
    )
    correction_args = SimpleNamespace(
        match_threshold=float(args.match_threshold),
        match_search_interval=int(args.match_search_interval),
        max_round=int(args.max_round),
        area_threshold=float(args.area_threshold),
        h_threshold=float(args.h_threshold),
        out_threshold=1.0,
        padding=int(args.csnet_padding),
        save_masks=False,
        enable_viz=False,
        minimal_output=True,
        dist_to_boundary=50,
    )
    csnet_work_dir = args.output_dir.resolve() / "work" / "csnet"
    csnet_work_dir.mkdir(parents=True, exist_ok=True)
    corrected, round_data, _ = run_multiround_correction(
        sctc,
        model,
        transforms,
        csnet_work_dir,
        correction_args,
    )
    save_sctc_atomic(corrected, csnet_path, datasets_dir)
    pd.DataFrame(round_data).to_csv(
        csnet_path.parent / "csnet_correction_rounds.csv",
        index=False,
    )
    export_corrected_masks(
        corrected,
        image_dataset,
        args.output_dir.resolve() / "livecellx_corrected_masks",
    )
    log(f"[CS-Net] saved: {csnet_path}")
    del corrected, sctc, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return csnet_path


def ensure_corrected_masks(
    csnet_path: Path,
    corrected_mask_dir: Path,
    image_dataset,
) -> None:
    """Export corrected masks when a cached CS-Net SCTC is being reused."""
    existing = list(corrected_mask_dir.glob("*.png")) if corrected_mask_dir.exists() else []
    if existing:
        return
    log(f"[CS-Net] corrected-mask export is missing; rebuilding {corrected_mask_dir}")
    sctc = SingleCellTrajectoryCollection.load_from_json_file(str(csnet_path), parallel=False)
    export_corrected_masks(sctc, image_dataset, corrected_mask_dir)
    del sctc
    gc.collect()


def run_corrected_mask_sort_stage(
    csnet_path: Path,
    corrected_mask_dir: Path,
    corrected_sort_path: Path,
    datasets_dir: Path,
    image_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, object, list]:
    """Rebuild trajectory identities by running SORT on CS-Net output masks."""
    corrected_image_dataset, corrected_mask_dataset = load_datasets(
        image_dir,
        corrected_mask_dir,
    )
    corrected_cells = prep_scs_from_mask_dataset(
        corrected_mask_dataset,
        corrected_image_dataset,
    )
    force = bool(
        args.force_corrected_sort or args.force_csnet or args.force_patches
    )
    if not output_is_stale(corrected_sort_path, csnet_path, force):
        log(f"[corrected SORT] reusing {corrected_sort_path}")
        return corrected_sort_path, corrected_image_dataset, corrected_cells

    log(
        "[corrected SORT] tracking CS-Net masks with "
        f"max_age={args.max_age}, min_hits={args.min_hits}"
    )
    corrected_sort = track_SORT_bbox_from_scs(
        corrected_cells,
        raw_imgs=corrected_image_dataset,
        mask_dataset=corrected_mask_dataset,
        max_age=int(args.max_age),
        min_hits=int(args.min_hits),
        sc_inplace=True,
    )
    save_sctc_atomic(corrected_sort, corrected_sort_path, datasets_dir)
    log(
        f"[corrected SORT] saved {len(corrected_sort)} trajectories and "
        f"{len(corrected_sort.get_all_scs())} cells: {corrected_sort_path}"
    )
    del corrected_sort
    gc.collect()
    return corrected_sort_path, corrected_image_dataset, corrected_cells


def timesformer_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        force_timesformer=bool(
            args.force_timesformer
            or args.force_patches
            or args.force_corrected_sort
            or args.force_csnet
        ),
        force_csnet=bool(
            args.force_patches or args.force_corrected_sort or args.force_csnet
        ),
        rerun_process1_csnet=False,
        timesformer_temp_root=str(
            (args.output_dir.resolve() / "work" / "timesformer_temp").resolve()
        ),
        timesformer_window_size=int(args.timesformer_window_size),
        timesformer_coarse_step_size=int(args.timesformer_coarse_step_size),
        timesformer_step_size=int(args.timesformer_step_size),
        timesformer_fine_context_cells=int(args.timesformer_fine_context_cells),
        timesformer_padding=int(args.timesformer_padding),
        timesformer_frame_type=str(args.timesformer_frame_type),
        timesformer_division_label=int(args.timesformer_division_label),
        timesformer_max_gap=int(args.timesformer_max_gap),
        timesformer_min_positive_windows=int(args.timesformer_min_positive_windows),
        daughter_search_window=int(args.daughter_search_window),
        daughter_max_distance=float(args.daughter_max_distance),
        timesformer_config=str(Path(args.timesformer_config).resolve()),
        timesformer_model=str(Path(args.timesformer_model).resolve()),
        timesformer_device=str(args.timesformer_device),
    )


def run_timesformer_stage(
    csnet_path: Path,
    final_path: Path,
    datasets_dir: Path,
    args: argparse.Namespace,
) -> Path:
    from result_process12_newtracking_fig4.run_process12_newtracking_fig4 import (
        apply_lineage_from_predictions,
        infer_all_trajectory_windows,
    )

    force = (
        args.force_timesformer
        or args.force_patches
        or args.force_corrected_sort
        or args.force_csnet
    )
    if not output_is_stale(final_path, csnet_path, force):
        log(f"[TimeSformer] reusing {final_path}")
        return final_path

    log(f"[TimeSformer] loading CS-Net collection: {csnet_path}")
    sctc = SingleCellTrajectoryCollection.load_from_json_file(str(csnet_path), parallel=False)
    tf_args = timesformer_namespace(args)
    predictions = infer_all_trajectory_windows(sctc, args.output_dir.resolve(), tf_args)
    apply_lineage_from_predictions(sctc, predictions, args.output_dir.resolve(), tf_args)
    save_sctc_atomic(sctc, final_path, datasets_dir)
    log(f"[TimeSformer] saved final lineage collection: {final_path}")
    del sctc
    gc.collect()
    return final_path


def validate_sctc(path: Path) -> dict:
    log(f"[validate] loading {path}")
    sctc = SingleCellTrajectoryCollection.load_from_json_file(str(path), parallel=False)
    result = {
        "path": str(path.resolve()),
        "trajectories": int(len(sctc)),
        "cells": int(len(sctc.get_all_scs())),
    }
    del sctc
    gc.collect()
    log(f"[validate] OK: {result['trajectories']} trajectories, {result['cells']} cells")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sort-sctc",
        type=Path,
        default=None,
        help="Existing SORT JSON; defaults to OUTPUT_DIR/traj_collections/sort.json.",
    )
    parser.add_argument("--csnet-model", type=Path, default=CSNET_MODEL)
    parser.add_argument("--csnet-device", default="cuda:0")
    parser.add_argument("--max-round", type=int, default=50)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--match-search-interval", type=int, default=3)
    parser.add_argument("--csnet-padding", type=int, default=20)
    parser.add_argument("--h-threshold", type=float, default=0.3)
    parser.add_argument("--area-threshold", type=float, default=1000)
    parser.add_argument("--max-age", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--timesformer-model", type=Path, default=TIMESFORMER_MODEL)
    parser.add_argument("--timesformer-config", type=Path, default=TIMESFORMER_CONFIG)
    parser.add_argument("--timesformer-device", default="cuda:0")
    parser.add_argument("--timesformer-window-size", type=int, default=8)
    parser.add_argument("--timesformer-coarse-step-size", type=int, default=8)
    parser.add_argument("--timesformer-step-size", type=int, default=1)
    parser.add_argument("--timesformer-fine-context-cells", type=int, default=8)
    parser.add_argument("--timesformer-padding", type=int, default=200)
    parser.add_argument(
        "--timesformer-frame-type",
        default="combined",
        choices=["video", "mask", "combined"],
    )
    parser.add_argument("--timesformer-division-label", type=int, default=0)
    parser.add_argument("--timesformer-max-gap", type=int, default=1)
    parser.add_argument("--timesformer-min-positive-windows", type=int, default=1)
    parser.add_argument("--daughter-search-window", type=int, default=8)
    parser.add_argument("--daughter-max-distance", type=float, default=180.0)
    parser.add_argument("--force-patches", action="store_true")
    parser.add_argument("--force-csnet", action="store_true")
    parser.add_argument("--force-corrected-sort", action="store_true")
    parser.add_argument("--force-timesformer", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.image_dir = args.image_dir.resolve()
    args.mask_dir = args.mask_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    trajectory_dir = args.output_dir / "traj_collections"
    datasets_dir = trajectory_dir / "datasets"
    sort_path = (
        args.sort_sctc.resolve()
        if args.sort_sctc is not None
        else trajectory_dir / "sort.json"
    )
    patched_path = trajectory_dir / "sort_post_patched.json"
    csnet_path = trajectory_dir / "livecellx_csnet.json"
    corrected_mask_dir = args.output_dir / "livecellx_corrected_masks"
    corrected_sort_path = trajectory_dir / "livecellx_corrected_sort.json"
    identity_repaired_path = (
        trajectory_dir / "livecellx_corrected_sort_identity_repaired.json"
    )
    corrected_patched_path = (
        trajectory_dir / "livecellx_corrected_sort_post_patched.json"
    )
    final_path = trajectory_dir / "livecellx_csnet_timesformer_lineage.json"

    require_path(args.image_dir, "image directory")
    require_path(args.mask_dir, "mask directory")
    require_path(sort_path, "existing SORT trajectory collection")
    require_path(args.csnet_model.resolve(), "CS-Net checkpoint")
    require_path(args.timesformer_model.resolve(), "TimeSformer checkpoint")
    require_path(args.timesformer_config.resolve(), "TimeSformer config")
    if args.max_age < 0:
        raise ValueError("--max-age cannot be negative")
    if args.min_hits <= 0:
        raise ValueError("--min-hits must be positive")
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    log("[LiveCellX] loading original images and masks for initial SORT patches")
    image_dataset, mask_dataset = load_datasets(args.image_dir, args.mask_dir)
    original_cells = prep_scs_from_mask_dataset(mask_dataset, image_dataset)
    run_post_sort_patches(
        sort_path,
        patched_path,
        datasets_dir,
        original_cells,
        force=args.force_patches,
        report_name="initial_sort_patch_report.json",
    )
    run_csnet_stage(
        patched_path,
        csnet_path,
        datasets_dir,
        image_dataset,
        args,
    )
    ensure_corrected_masks(csnet_path, corrected_mask_dir, image_dataset)
    _, _, corrected_cells = run_corrected_mask_sort_stage(
        csnet_path,
        corrected_mask_dir,
        corrected_sort_path,
        datasets_dir,
        args.image_dir,
        args,
    )
    run_post_sort_patches(
        corrected_sort_path,
        corrected_patched_path,
        datasets_dir,
        corrected_cells,
        force=(
            args.force_patches
            or args.force_csnet
            or args.force_corrected_sort
        ),
        report_name="corrected_sort_patch_report.json",
    )
    run_identity_switch_repair(
        corrected_patched_path,
        identity_repaired_path,
        datasets_dir,
        force=(
            args.force_patches
            or args.force_csnet
            or args.force_corrected_sort
        ),
    )
    run_timesformer_stage(identity_repaired_path, final_path, datasets_dir, args)

    outputs = {
        "sort": str(sort_path),
        "sort_post_patched": str(patched_path),
        "livecellx_csnet": str(csnet_path),
        "livecellx_corrected_sort": str(corrected_sort_path),
        "livecellx_corrected_sort_identity_repaired": str(identity_repaired_path),
        "livecellx_corrected_sort_post_patched": str(corrected_patched_path),
        "livecellx_final": str(final_path),
    }
    manifest = {
        "pipeline_order": [
            "existing_SORT",
            "restore_untracked_initial_cells",
            "reconnect_initial_trajectory_fragments",
            "CS-Net",
            "export_corrected_masks",
            "SORT_on_corrected_masks",
            "restore_untracked_corrected_cells",
            "reconnect_corrected_trajectory_fragments",
            "repair_corrected_SORT_identity_switches",
            "TimeSformer_lineage",
        ],
        "image_dir": str(args.image_dir),
        "mask_dir": str(args.mask_dir),
        "corrected_mask_dir": str(corrected_mask_dir),
        "sort_parameters": {
            "max_age": int(args.max_age),
            "min_hits": int(args.min_hits),
        },
        "outputs": outputs,
        "csnet_model": str(args.csnet_model.resolve()),
        "timesformer_model": str(args.timesformer_model.resolve()),
    }
    if not args.skip_validation:
        manifest["validation"] = {
            name: validate_sctc(Path(path)) for name, path in outputs.items()
        }
    with (trajectory_dir / "livecellx_pipeline_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    log(f"[LiveCellX] final trajectory collection: {final_path}")


if __name__ == "__main__":
    main()
