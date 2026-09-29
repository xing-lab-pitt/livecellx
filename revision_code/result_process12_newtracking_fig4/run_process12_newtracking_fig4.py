#!/usr/bin/env python
"""Build matched-parameter SORT and LiveCellX collections for process1/2.

This pipeline creates two full trajectory collections for each dataset:

1. SORT, using ``max_age=5`` and ``min_hits=1``.
2. LiveCellX, starting from the same SORT result, applying CS-Net mask
   correction, and then applying TimeSformer-based mitosis/lineage correction.

TimeSformer inference does not use ground truth. Every eligible CS-Net
trajectory is first screened with non-overlapping eight-frame clips. Only
coarse-positive regions are then refined with a one-frame sliding step before
the original Process2 split and daughter-search logic is applied. Per-
trajectory predictions are written atomically, so rerunning this command
resumes unfinished TimeSformer work.

By default, the existing process1 CS-Net result is reused only after verifying
that its saved metadata records max_age=5 and min_hits=1. Process2 is rebuilt
from the original images and Cellpose masks because its earlier tracking used
min_hits=3. Use ``--rerun-process1-csnet`` to rebuild process1 as well.

All final SCTC files are written with absolute dataset-JSON references and are
loaded once at the end of the run to verify that they can be opened directly by
``SingleCellTrajectoryCollection.load_from_json_file``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from livecellx.core.single_cell import SingleCellTrajectory, SingleCellTrajectoryCollection


SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_DIR = SCRIPT_DIR.parent
REPO_ROOT = REVISION_DIR.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(REVISION_DIR))

CSNET_MODEL = REVISION_DIR / "model" / "last.ckpt"
TIMESFORMER_MODEL = REVISION_DIR / "model" / "best_acc_top1_epoch_65.pth"
TIMESFORMER_CONFIG = (
    REVISION_DIR / "configs" / "timesformer_divst_v15.py"
)

MAX_AGE = 5
MIN_HITS = 1
SORT_FILENAME = "sort_max_age5_min_hits1.json"
LIVECELLX_FILENAME = "livecellx_csnet_timesformer_coarse_to_fine_max_age5_min_hits1.json"
COMPLETED_PROCESS1_LIVECELLX_FILENAME = "livecellx_csnet_timesformer_max_age5_min_hits1.json"
CSNET_ONLY_FILENAME = "livecellx_csnet_only_max_age5_min_hits1.json"
TIMESFORMER_INFERENCE_VERSION = "coarse_to_fine_v1"

PREDICTION_COLUMNS = [
    "track_id",
    "start_time",
    "end_time",
    "start_index",
    "end_index",
    "inference_stage",
    "video_path",
    "pred_label",
    "division_score",
    "is_division",
    "error",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    image_dir: Path
    mask_dir: Path
    reusable_sort: Optional[Path] = None
    reusable_csnet: Optional[Path] = None
    reusable_metrics: Optional[Path] = None
    reusable_datasets: Optional[Path] = None


DATASETS: Dict[str, DatasetSpec] = {
    "process1": DatasetSpec(
        name="process1",
        image_dir=REVISION_DIR / "data" / "process1_30den" / "2D",
        mask_dir=REVISION_DIR / "data" / "process1_30den" / "2D_mask",
        reusable_sort=REVISION_DIR / "results_process1_csnet" / "livecellx_sctc_before.json",
        reusable_csnet=REVISION_DIR / "results_process1_csnet" / "livecellx_sctc_after.json",
        reusable_metrics=REVISION_DIR / "results_process1_csnet" / "livecellx_metrics.json",
        reusable_datasets=REVISION_DIR / "results_process1_csnet" / "datasets",
    ),
    "process2": DatasetSpec(
        name="process2",
        image_dir=REVISION_DIR / "data" / "process2" / "2D",
        mask_dir=REVISION_DIR / "data" / "process2" / "2D_mask",
    ),
}


def log(message: str) -> None:
    print(message, flush=True)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def atomic_write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(data, handle, indent=2)
    os.replace(tmp_path, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def copy_dataset_jsons(source_dir: Path, target_dir: Path) -> None:
    require_path(source_dir, "source dataset metadata directory")
    target_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(source_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No dataset JSON files found in {source_dir}")
    for source in files:
        shutil.copy2(source, target_dir / source.name)


def stream_copy_with_replacements(source: Path, target: Path, replacements: Dict[str, str]) -> None:
    """Copy a potentially multi-GB JSON while replacing dataset path strings."""
    require_path(source, "source trajectory collection")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    byte_replacements = {
        old.encode("utf-8"): new.encode("utf-8")
        for old, new in replacements.items()
        if old and old != new
    }
    with source.open("rb") as src, tmp_path.open("wb") as dst:
        for line in src:
            for old, new in byte_replacements.items():
                line = line.replace(old, new)
            dst.write(line)
    os.replace(tmp_path, target)


def dataset_path_replacements(source_dataset_dir: Path, target_dataset_dir: Path) -> Dict[str, str]:
    target = str(target_dataset_dir.resolve())
    legacy_results = str((REVISION_DIR / "results" / "datasets").resolve())
    replacements = {
        str(source_dataset_dir.resolve()): target,
        legacy_results: target,
        "revision_code/results/datasets": target,
        "revision_code/results_process1_csnet/datasets": target,
        "revision_code/results_process2_csnet/datasets": target,
    }
    return replacements


def verify_process1_reuse_metadata(spec: DatasetSpec) -> None:
    if spec.reusable_metrics is None:
        raise ValueError("No process1 reusable metrics path configured")
    require_path(spec.reusable_metrics, "process1 metrics metadata")
    with spec.reusable_metrics.open() as handle:
        metrics = json.load(handle)
    config = metrics.get("config", {})
    max_age = int(config.get("max_age", -1))
    min_hits = int(config.get("min_hits", -1))
    if (max_age, min_hits) != (MAX_AGE, MIN_HITS):
        raise ValueError(
            "Refusing to reuse process1 CS-Net files: saved metadata reports "
            f"max_age={max_age}, min_hits={min_hits}, expected {MAX_AGE}/{MIN_HITS}."
        )
    log(f"[process1] verified reusable tracking parameters: max_age={max_age}, min_hits={min_hits}")


def prepare_reused_process1(spec: DatasetSpec, dataset_dir: Path, force: bool) -> Path:
    if not all([spec.reusable_sort, spec.reusable_csnet, spec.reusable_datasets]):
        raise ValueError("Incomplete reusable process1 configuration")
    verify_process1_reuse_metadata(spec)
    require_path(spec.reusable_sort, "reusable process1 SORT collection")
    require_path(spec.reusable_csnet, "reusable process1 CS-Net collection")

    datasets_dir = dataset_dir / "datasets"
    sort_path = dataset_dir / SORT_FILENAME
    csnet_path = dataset_dir / "work" / CSNET_ONLY_FILENAME
    if force or not sort_path.exists() or not csnet_path.exists():
        copy_dataset_jsons(spec.reusable_datasets, datasets_dir)
        replacements = dataset_path_replacements(spec.reusable_datasets, datasets_dir)
        log(f"[process1] copying portable SORT collection to {sort_path}")
        stream_copy_with_replacements(spec.reusable_sort, sort_path, replacements)
        log(f"[process1] copying portable CS-Net collection to {csnet_path}")
        stream_copy_with_replacements(spec.reusable_csnet, csnet_path, replacements)
    else:
        log("[process1] reusing prepared SORT and CS-Net-only collections")
    return csnet_path


def run_csnet_from_scratch(spec: DatasetSpec, dataset_dir: Path, args: argparse.Namespace) -> Path:
    work_dir = dataset_dir / "work" / "csnet_pipeline"
    raw_sort = work_dir / "livecellx_sctc_before.json"
    raw_csnet = work_dir / "livecellx_sctc_after.json"
    sort_path = dataset_dir / SORT_FILENAME
    csnet_path = dataset_dir / "work" / CSNET_ONLY_FILENAME
    rebuild_stage = args.force_csnet or (spec.name == "process1" and args.rerun_process1_csnet)

    if rebuild_stage:
        for path in [raw_sort, raw_csnet, sort_path, csnet_path]:
            if path.exists():
                path.unlink()

    if not raw_sort.exists() or not raw_csnet.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            str(REVISION_DIR / "run_livecellx.py"),
            "--image_dir",
            str(spec.image_dir.resolve()),
            "--mask_dir",
            str(spec.mask_dir.resolve()),
            "--model_ckpt",
            str(Path(args.csnet_model).resolve()),
            "--output_dir",
            str(work_dir.resolve()),
            "--max_age",
            str(MAX_AGE),
            "--min_hits",
            str(MIN_HITS),
        ]
        log(f"[{spec.name}] running SORT(5/1) + CS-Net")
        log("[command] " + " ".join(command))
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    else:
        log(f"[{spec.name}] reusing completed SORT(5/1) + CS-Net stage")

    require_path(raw_sort, f"{spec.name} SORT stage output")
    require_path(raw_csnet, f"{spec.name} CS-Net stage output")
    source_datasets = work_dir / "datasets"
    target_datasets = dataset_dir / "datasets"
    copy_dataset_jsons(source_datasets, target_datasets)
    replacements = dataset_path_replacements(source_datasets, target_datasets)
    if not sort_path.exists() or rebuild_stage:
        log(f"[{spec.name}] writing portable SORT collection to {sort_path}")
        stream_copy_with_replacements(raw_sort, sort_path, replacements)
    if not csnet_path.exists() or rebuild_stage:
        log(f"[{spec.name}] writing portable CS-Net-only collection to {csnet_path}")
        stream_copy_with_replacements(raw_csnet, csnet_path, replacements)
    return csnet_path


def prepare_sort_and_csnet(spec: DatasetSpec, dataset_dir: Path, args: argparse.Namespace) -> Path:
    if spec.name == "process1" and not args.rerun_process1_csnet:
        return prepare_reused_process1(spec, dataset_dir, force=args.force_csnet)
    return run_csnet_from_scratch(spec, dataset_dir, args)


def timesformer_namespace(args: argparse.Namespace, video_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        timesformer_window_size=int(args.timesformer_window_size),
        timesformer_step_size=int(args.timesformer_step_size),
        timesformer_padding=int(args.timesformer_padding),
        timesformer_frame_type=str(args.timesformer_frame_type),
        timesformer_division_label=int(args.timesformer_division_label),
        timesformer_max_gap=int(args.timesformer_max_gap),
        timesformer_min_positive_windows=int(args.timesformer_min_positive_windows),
        daughter_search_window=int(args.daughter_search_window),
        daughter_max_distance=float(args.daughter_max_distance),
        timesformer_video_dir=str(video_dir),
    )


def prediction_csv_path(prediction_dir: Path, track_id: int) -> Path:
    return prediction_dir / f"trajectory_{int(track_id):06d}.csv"


def done_path(prediction_dir: Path, track_id: int) -> Path:
    return prediction_dir / f"trajectory_{int(track_id):06d}.done.json"


def trajectory_subset(
    traj: SingleCellTrajectory,
    first_index: int,
    last_index: int,
) -> SingleCellTrajectory:
    sorted_scs = traj.get_sorted_scs()
    selected = sorted_scs[int(first_index) : int(last_index) + 1]
    return SingleCellTrajectory(
        track_id=int(traj.track_id),
        timeframe_to_single_cell={int(sc.timeframe): sc for sc in selected},
        img_dataset=traj.img_dataset,
        mask_dataset=traj.mask_dataset,
        extra_datasets=traj.extra_datasets,
        mother_trajectories=set(),
        daughter_trajectories=set(),
    )


def annotate_prediction_stage(
    pred_df: pd.DataFrame,
    traj: SingleCellTrajectory,
    stage: str,
) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    time_to_index = {
        int(sc.timeframe): index
        for index, sc in enumerate(traj.get_sorted_scs())
    }
    pred_df = pred_df.copy()
    pred_df["start_index"] = pred_df["start_time"].astype(int).map(time_to_index)
    pred_df["end_index"] = pred_df["end_time"].astype(int).map(time_to_index)
    if pred_df[["start_index", "end_index"]].isna().any().any():
        raise RuntimeError(
            f"Could not map TimeSformer windows for track {int(traj.track_id)}"
        )
    pred_df["start_index"] = pred_df["start_index"].astype(int)
    pred_df["end_index"] = pred_df["end_index"].astype(int)
    pred_df["inference_stage"] = str(stage)
    return pred_df.reindex(columns=PREDICTION_COLUMNS)


def fine_candidate_intervals(
    coarse_df: pd.DataFrame,
    trajectory_length: int,
    context_cells: int,
) -> List[tuple]:
    if coarse_df.empty:
        return []
    positives = coarse_df[coarse_df["is_division"].astype(bool)].sort_values("start_index")
    intervals: List[tuple] = []
    for row in positives.to_dict("records"):
        start = max(0, int(row["start_index"]) - int(context_cells))
        end = min(int(trajectory_length) - 1, int(row["end_index"]) + int(context_cells))
        if intervals and start <= intervals[-1][1] + 1:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return intervals


def infer_coarse_to_fine_for_traj(
    traj: SingleCellTrajectory,
    model,
    trajectory_temp: Path,
    args: argparse.Namespace,
) -> tuple:
    """Run the original Process2 inference on coarse windows, then positive regions."""
    from process2_ctc_style_tracking_evaluation import infer_timesformer_windows_for_traj

    coarse_dir = trajectory_temp / "coarse"
    coarse_args = timesformer_namespace(args, coarse_dir)
    coarse_args.timesformer_step_size = int(args.timesformer_coarse_step_size)
    coarse_raw = infer_timesformer_windows_for_traj(traj, model, coarse_dir, coarse_args)
    coarse_df = annotate_prediction_stage(coarse_raw, traj, "coarse")

    coarse_errors = coarse_df.get("error", pd.Series(dtype=str)).fillna("").astype(str)
    if (coarse_errors.str.len() > 0).any():
        first_error = coarse_df.loc[coarse_errors.str.len() > 0, "error"].iloc[0]
        raise RuntimeError(f"Coarse TimeSformer failed for track {int(traj.track_id)}: {first_error}")

    intervals = fine_candidate_intervals(
        coarse_df,
        len(traj),
        int(args.timesformer_fine_context_cells),
    )
    fine_frames: List[pd.DataFrame] = []
    for interval_index, (first_index, last_index) in enumerate(intervals):
        subset = trajectory_subset(traj, first_index, last_index)
        if len(subset) < int(args.timesformer_window_size):
            continue
        fine_dir = trajectory_temp / f"fine_{interval_index:03d}"
        fine_args = timesformer_namespace(args, fine_dir)
        fine_args.timesformer_step_size = int(args.timesformer_step_size)
        fine_raw = infer_timesformer_windows_for_traj(subset, model, fine_dir, fine_args)
        fine_df = annotate_prediction_stage(fine_raw, traj, "fine")
        if len(fine_df):
            fine_frames.append(fine_df)

    fine_df = (
        pd.concat(fine_frames, ignore_index=True)
        if fine_frames
        else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    if len(fine_df):
        fine_df = fine_df.drop_duplicates(subset=["start_time", "end_time"], keep="first")
        fine_errors = fine_df.get("error", pd.Series(dtype=str)).fillna("").astype(str)
        if (fine_errors.str.len() > 0).any():
            first_error = fine_df.loc[fine_errors.str.len() > 0, "error"].iloc[0]
            raise RuntimeError(f"Fine TimeSformer failed for track {int(traj.track_id)}: {first_error}")

    combined = pd.concat([coarse_df, fine_df], ignore_index=True)
    summary = {
        "coarse_windows": int(len(coarse_df)),
        "coarse_positive_windows": int(coarse_df["is_division"].astype(bool).sum()) if len(coarse_df) else 0,
        "fine_intervals": int(len(intervals)),
        "fine_windows": int(len(fine_df)),
    }
    return combined.reindex(columns=PREDICTION_COLUMNS), summary


def infer_all_trajectory_windows(
    sctc: SingleCellTrajectoryCollection,
    dataset_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    from process2_ctc_style_tracking_evaluation import load_timesformer_model

    prediction_root = dataset_dir / "work" / f"timesformer_predictions_{TIMESFORMER_INFERENCE_VERSION}"
    prediction_dir = prediction_root / "per_trajectory"
    temporary_root = Path(args.timesformer_temp_root).resolve() / dataset_dir.name

    if args.force_timesformer or args.force_csnet or (dataset_dir.name == "process1" and args.rerun_process1_csnet):
        shutil.rmtree(prediction_root, ignore_errors=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir(parents=True, exist_ok=True)

    process_ids = sorted(int(tid) for tid in sctc.get_track_ids())
    incomplete_ids = [tid for tid in process_ids if not done_path(prediction_dir, tid).exists()]
    log(
        f"[{dataset_dir.name}] TimeSformer coarse-to-fine: {len(process_ids)} total, "
        f"{len(incomplete_ids)} unfinished"
    )

    model = None
    if any(len(sctc.get_trajectory(tid)) >= int(args.timesformer_window_size) for tid in incomplete_ids):
        model = load_timesformer_model(
            Path(args.timesformer_config),
            Path(args.timesformer_model),
            str(args.timesformer_device),
            int(args.timesformer_window_size),
        )

    try:
        for index, tid in enumerate(incomplete_ids, start=1):
            traj = sctc.get_trajectory(tid)
            csv_path = prediction_csv_path(prediction_dir, tid)
            marker_path = done_path(prediction_dir, tid)
            trajectory_temp = temporary_root / f"trajectory_{tid:06d}"
            shutil.rmtree(trajectory_temp, ignore_errors=True)
            log(
                f"[{dataset_dir.name}] coarse-to-fine {index}/{len(incomplete_ids)}: "
                f"track {tid}, cells={len(traj)}"
            )
            try:
                if len(traj) < int(args.timesformer_window_size):
                    pred_df = pd.DataFrame(columns=PREDICTION_COLUMNS)
                    status = "too_short"
                    stage_summary = {
                        "coarse_windows": 0,
                        "coarse_positive_windows": 0,
                        "fine_intervals": 0,
                        "fine_windows": 0,
                    }
                else:
                    pred_df, stage_summary = infer_coarse_to_fine_for_traj(
                        traj, model, trajectory_temp, args
                    )
                    status = "complete"
                    log(
                        f"[{dataset_dir.name}] track {tid}: "
                        f"coarse={stage_summary['coarse_windows']}, "
                        f"positive={stage_summary['coarse_positive_windows']}, "
                        f"fine={stage_summary['fine_windows']}"
                    )

                atomic_write_csv(pred_df.reindex(columns=PREDICTION_COLUMNS), csv_path)
                atomic_write_json(
                    {
                        "inference_version": TIMESFORMER_INFERENCE_VERSION,
                        "track_id": int(tid),
                        "trajectory_cells": int(len(traj)),
                        "prediction_windows": int(len(pred_df)),
                        "status": status,
                        **stage_summary,
                    },
                    marker_path,
                )
            finally:
                shutil.rmtree(trajectory_temp, ignore_errors=True)
    finally:
        del model
        gc.collect()

    frames: List[pd.DataFrame] = []
    missing_markers = []
    for tid in process_ids:
        marker = done_path(prediction_dir, tid)
        csv_path = prediction_csv_path(prediction_dir, tid)
        if not marker.exists() or not csv_path.exists():
            missing_markers.append(tid)
            continue
        frame = pd.read_csv(csv_path)
        if len(frame):
            frames.append(frame)
    if missing_markers:
        raise RuntimeError(f"Missing coarse-to-fine predictions for tracks: {missing_markers[:20]}")
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PREDICTION_COLUMNS)
    atomic_write_csv(predictions, dataset_dir / "timesformer_window_predictions.csv")
    atomic_write_csv(
        predictions,
        dataset_dir / f"timesformer_window_predictions_{TIMESFORMER_INFERENCE_VERSION}.csv",
    )
    return predictions


def apply_lineage_from_predictions(
    sctc: SingleCellTrajectoryCollection,
    predictions: pd.DataFrame,
    dataset_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    from process2_ctc_style_tracking_evaluation import (
        add_lineage_relation,
        choose_split_from_predictions,
        find_second_daughter,
        split_trajectory_after_time,
        summarize_lineage_relation_consistency,
    )

    helper_args = timesformer_namespace(args, Path(args.timesformer_temp_root))
    if len(predictions):
        predictions = predictions.copy()
        predictions["track_id"] = predictions["track_id"].astype(int)
    original_ids = sorted(int(tid) for tid in sctc.get_track_ids())
    next_tid = int(sctc.get_max_tid()) + 1
    rows = []

    for tid in original_ids:
        traj = sctc.get_trajectory(tid)
        track_predictions = (
            predictions[predictions["track_id"] == tid].copy()
            if len(predictions)
            else pd.DataFrame(columns=PREDICTION_COLUMNS)
        )
        fine_predictions = (
            track_predictions[
                track_predictions["inference_stage"].astype(str) == "fine"
            ].copy()
            if len(track_predictions) and "inference_stage" in track_predictions
            else pd.DataFrame(columns=PREDICTION_COLUMNS)
        )
        # Coarse positives select candidate regions; step=1 predictions locate
        # the division using the unchanged Process2 split-selection logic.
        pred_df = fine_predictions if len(fine_predictions) else track_predictions
        split_info = choose_split_from_predictions(pred_df, helper_args)
        if split_info is None:
            rows.append(
                {
                    "source_track_id": tid,
                    "status": "no_mitosis_window",
                    "split_t": np.nan,
                    "daughter_a_track_id": np.nan,
                    "daughter_b_track_id": np.nan,
                    "daughter_b_score": np.nan,
                    "daughter_b_match_time": np.nan,
                }
            )
            continue

        split_t = int(split_info["split_t"])
        daughter_a = split_trajectory_after_time(sctc, traj, split_t, next_tid)
        if daughter_a is None:
            rows.append(
                {
                    "source_track_id": tid,
                    "status": "split_outside_trajectory",
                    **split_info,
                    "daughter_a_track_id": np.nan,
                    "daughter_b_track_id": np.nan,
                    "daughter_b_score": np.nan,
                    "daughter_b_match_time": np.nan,
                }
            )
            continue

        next_tid += 1
        daughter_b, daughter_b_score, daughter_b_time = find_second_daughter(
            sctc,
            traj,
            daughter_a,
            split_t,
            int(args.daughter_search_window),
            float(args.daughter_max_distance),
        )
        add_lineage_relation(traj, daughter_a)
        add_lineage_relation(traj, daughter_b)
        rows.append(
            {
                "source_track_id": tid,
                "status": "corrected_with_two_daughters" if daughter_b is not None else "corrected_one_daughter_only",
                **split_info,
                "daughter_a_track_id": int(daughter_a.track_id),
                "daughter_b_track_id": int(daughter_b.track_id) if daughter_b is not None else np.nan,
                "daughter_b_score": daughter_b_score,
                "daughter_b_match_time": daughter_b_time,
            }
        )

    corrections = pd.DataFrame(rows)
    atomic_write_csv(corrections, dataset_dir / "timesformer_lineage_corrections.csv")
    consistency = summarize_lineage_relation_consistency(sctc)
    serializable_consistency = {
        key: value for key, value in consistency.items() if key != "duplicate_daughter_to_mothers"
    }
    atomic_write_json(serializable_consistency, dataset_dir / "timesformer_lineage_consistency.json")
    duplicate_rows = [
        {"daughter_track_id": tid, "mother_track_ids": ",".join(str(x) for x in mother_ids)}
        for tid, mother_ids in consistency["duplicate_daughter_to_mothers"].items()
    ]
    atomic_write_csv(pd.DataFrame(duplicate_rows), dataset_dir / "timesformer_duplicate_daughters.csv")
    log(
        f"[{dataset_dir.name}] TimeSformer lineage: "
        f"{consistency['n_mother_tracks_with_daughters']} mothers, "
        f"{consistency['n_mother_to_daughter_edges']} edges, "
        f"{consistency['n_duplicate_daughter_tracks']} duplicate daughter assignments"
    )
    return corrections


def run_timesformer_stage(csnet_path: Path, dataset_dir: Path, args: argparse.Namespace) -> Path:
    can_reuse = (
        not args.force_timesformer
        and not args.force_csnet
        and not (dataset_dir.name == "process1" and args.rerun_process1_csnet)
    )
    if dataset_dir.name == "process1" and can_reuse:
        completed_process1 = dataset_dir / COMPLETED_PROCESS1_LIVECELLX_FILENAME
        if completed_process1.exists():
            log(
                "[process1] reusing the already completed full step=1 "
                f"TimeSformer result: {completed_process1}"
            )
            return completed_process1

    final_path = dataset_dir / LIVECELLX_FILENAME
    if final_path.exists() and can_reuse:
        log(f"[{dataset_dir.name}] reusing completed LiveCellX final collection: {final_path}")
        return final_path

    log(f"[{dataset_dir.name}] loading CS-Net-only collection: {csnet_path}")
    sctc = SingleCellTrajectoryCollection.load_from_json_file(str(csnet_path), parallel=False)
    predictions = infer_all_trajectory_windows(sctc, dataset_dir, args)
    apply_lineage_from_predictions(sctc, predictions, dataset_dir, args)

    datasets_dir = dataset_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    log(f"[{dataset_dir.name}] writing final LiveCellX collection: {final_path}")
    sctc.write_json(str(tmp_path), dataset_json_dir=datasets_dir.resolve())
    os.replace(tmp_path, final_path)
    del sctc
    gc.collect()
    return final_path


def validate_collection(path: Path) -> Dict[str, int]:
    require_path(path, "final trajectory collection")
    log(f"[validate] loading {path}")
    sctc = SingleCellTrajectoryCollection.load_from_json_file(str(path), parallel=False)
    summary = {
        "trajectories": int(len(sctc)),
        "cells": int(len(sctc.get_all_scs())),
    }
    del sctc
    gc.collect()
    log(f"[validate] OK: {summary['trajectories']} trajectories, {summary['cells']} cells")
    return summary


def write_manifest(output_root: Path, summaries: Dict[str, dict], args: argparse.Namespace) -> None:
    manifest = {
        "tracking_parameters": {"max_age": MAX_AGE, "min_hits": MIN_HITS, "sort_iou_threshold": 0.3},
        "csnet_model": str(Path(args.csnet_model).resolve()),
        "timesformer_model": str(Path(args.timesformer_model).resolve()),
        "timesformer_config": str(Path(args.timesformer_config).resolve()),
        "timesformer_uses_ground_truth": False,
        "timesformer_inference": {
            "version": TIMESFORMER_INFERENCE_VERSION,
            "window_size": int(args.timesformer_window_size),
            "coarse_step_size": int(args.timesformer_coarse_step_size),
            "fine_step_size": int(args.timesformer_step_size),
            "fine_context_cells": int(args.timesformer_fine_context_cells),
            "frame_type": str(args.timesformer_frame_type),
        },
        "datasets": summaries,
    }
    atomic_write_json(manifest, output_root / "run_manifest.json")


def validate_inputs(args: argparse.Namespace) -> None:
    require_path(Path(args.csnet_model), "CS-Net checkpoint")
    require_path(Path(args.timesformer_model), "TimeSformer checkpoint")
    require_path(Path(args.timesformer_config), "TimeSformer config")
    require_path(REVISION_DIR / "run_livecellx.py", "CS-Net pipeline script")
    if int(args.timesformer_window_size) <= 0:
        raise ValueError("--timesformer-window-size must be positive")
    if int(args.timesformer_coarse_step_size) <= 0:
        raise ValueError("--timesformer-coarse-step-size must be positive")
    if int(args.timesformer_step_size) <= 0:
        raise ValueError("--timesformer-step-size must be positive")
    if int(args.timesformer_fine_context_cells) < 0:
        raise ValueError("--timesformer-fine-context-cells cannot be negative")
    for name in args.datasets:
        spec = DATASETS[name]
        require_path(spec.image_dir, f"{name} image directory")
        require_path(spec.mask_dir, f"{name} mask directory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["process1", "process2"])
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--csnet-model", default=str(CSNET_MODEL))
    parser.add_argument("--timesformer-model", default=str(TIMESFORMER_MODEL))
    parser.add_argument("--timesformer-config", default=str(TIMESFORMER_CONFIG))
    parser.add_argument("--timesformer-device", default="cuda:0")
    parser.add_argument("--timesformer-window-size", type=int, default=8)
    parser.add_argument(
        "--timesformer-coarse-step-size",
        type=int,
        default=8,
        help="Step for full-trajectory coarse screening (default: 8 frames).",
    )
    parser.add_argument(
        "--timesformer-step-size",
        type=int,
        default=1,
        help="Step for local fine screening around coarse-positive clips.",
    )
    parser.add_argument(
        "--timesformer-fine-context-cells",
        type=int,
        default=8,
        help="Cells added on each side of a coarse-positive region for refinement.",
    )
    parser.add_argument("--timesformer-padding", type=int, default=200)
    parser.add_argument("--timesformer-frame-type", default="combined", choices=["video", "mask", "combined"])
    parser.add_argument("--timesformer-division-label", type=int, default=0)
    parser.add_argument("--timesformer-max-gap", type=int, default=1)
    parser.add_argument("--timesformer-min-positive-windows", type=int, default=1)
    parser.add_argument("--daughter-search-window", type=int, default=8)
    parser.add_argument("--daughter-max-distance", type=float, default=180.0)
    parser.add_argument("--timesformer-temp-root", default="/tmp/livecellx_process12_timesformer")
    parser.add_argument(
        "--rerun-process1-csnet",
        action="store_true",
        help="Rebuild process1 SORT and CS-Net instead of reusing the verified existing 5/1 result.",
    )
    parser.add_argument("--force-csnet", action="store_true", help="Delete and rebuild prepared SORT/CS-Net files.")
    parser.add_argument(
        "--force-timesformer",
        action="store_true",
        help="Discard saved per-trajectory TimeSformer predictions and rebuild the final LiveCellX file.",
    )
    parser.add_argument(
        "--skip-final-validation",
        action="store_true",
        help="Skip full load-back validation of the four multi-GB output collections.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    validate_inputs(args)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, dict] = {}
    for name in args.datasets:
        spec = DATASETS[name]
        dataset_dir = output_root / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        log("=" * 80)
        log(f"[{name}] SORT(5/1) -> CS-Net -> TimeSformer")
        log("=" * 80)

        csnet_path = prepare_sort_and_csnet(spec, dataset_dir, args)
        sort_path = dataset_dir / SORT_FILENAME
        livecellx_path = run_timesformer_stage(csnet_path, dataset_dir, args)

        # Match the established final-results convention used by the Napari
        # editor: the first existing source frame is index 0. Reindex both
        # collections and their shared datasets only after inference, so the
        # tracking and TimeSformer logic still sees original frame numbers.
        from reindex_sctc_zero_based import reindex_outputs

        reindex_outputs(
            [sort_path, livecellx_path],
            dataset_dir / "datasets",
        )

        dataset_summary = {
            "sort_path": str(sort_path),
            "livecellx_path": str(livecellx_path),
        }
        if not args.skip_final_validation:
            dataset_summary["sort"] = validate_collection(sort_path)
            dataset_summary["livecellx"] = validate_collection(livecellx_path)
        summaries[name] = dataset_summary
        write_manifest(output_root, summaries, args)

    write_manifest(output_root, summaries, args)
    log("=" * 80)
    log(f"Completed. Results: {output_root}")
    for name, info in summaries.items():
        log(f"[{name}] SORT:      {info['sort_path']}")
        log(f"[{name}] LiveCellX: {info['livecellx_path']}")


if __name__ == "__main__":
    main()
