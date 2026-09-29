#!/usr/bin/env python
"""CTC-style tracking comparison on the process2 hand-annotated GT subset.

The official Cell Tracking Challenge scores require complete gold annotations so
that false positives and AOGM operations can be counted over the full movie.  The
process2 GT used here is intentionally partial: it contains the hand-annotated
division trajectories.  Therefore this script computes CTC-style quantities on
that annotated subset only:

* GT-region DET: fraction of annotated GT cells detected with IoU >= threshold.
* GT-region SEG: mean IoU over annotated GT cells, with missed cells
  contributing zero.
* Subgraph LNK: fraction of consecutive annotated GT trajectory links recovered
  by the same target track identity.
* Subgraph TF: mean longest continuously recovered fraction per annotated GT
  trajectory.
* Subgraph CT: fraction of annotated GT trajectories recovered completely inside
  the annotated span.
* Subgraph BC-anchor(i): fraction of annotated division events whose mother-last
  and two daughter-start anchors are recovered as three distinct target tracks,
  allowing i-frame GT-anchor tolerance.

The plotted target conditions are:
    SORT
    LiveCellX
    Ultrack

For LiveCellX, GT is not used to create the lineage relation.
The script optionally runs the TimeSformer mitosis classifier on each after-CSNet
trajectory, splits a trajectory at the estimated division point, chooses the
second daughter from nearby post-division trajectories, and then evaluates that
corrected collection against the partial GT.

Results are written to revision_code/latest_results/figures by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-livecellx")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from livecellx.core.single_cell import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results_process2_csnet"
LATEST_RESULTS_DIR = SCRIPT_DIR / "latest_results"
FIG_DIR = LATEST_RESULTS_DIR / "figures"
FINAL_TRAJ_COLLECTION_DIR = RESULTS_DIR / "final_results_traj_collections"

GT_PATH = (
    FINAL_TRAJ_COLLECTION_DIR
    / "sctc-final-2026-06-10_100_corrected_motherdaughter_corrected_gt_region_only_missing_fixed.json"
)
SORT_BEFORE_PATH = (
    FINAL_TRAJ_COLLECTION_DIR
    / "sctc-final-2026-0610_before_livecellx_gt_region_only.json"
)
SORT_BEFORE_FALLBACK_PATH = SORT_BEFORE_PATH
SORT_AFTER_PATH = (
    FINAL_TRAJ_COLLECTION_DIR
    / "sctc-final-2026-06-02_before_correction_timesformer_lineage_corrected_gt_region_only.json"
)
ULTRACK_EXP4_ZARR_PATH = SCRIPT_DIR / "tracking_from_cellpose_process2_beforecsnet_exp4"
ULTRACK_EXP4_GT_REGION_SCTC_PATH = (
    FINAL_TRAJ_COLLECTION_DIR / "ultrack_from_cellpose_process2_beforecsnet_exp4_gt_region_only.json"
)
ULTRACK_PATH = ULTRACK_EXP4_GT_REGION_SCTC_PATH
ULTRACK_ZARR_PATH = ULTRACK_PATH  # Backward-compatible arg name; default now uses the requested exp4 GT-region SCTC.
TIMESFORMER_CONFIG_PATH = (
    SCRIPT_DIR / "configs" / "timesformer_divst_v15.py"
)
TIMESFORMER_MODEL_PATH = SCRIPT_DIR / "model" / "best_acc_top1_epoch_65.pth"
TIMESFORMER_CORRECTED_AFTER_PATH = SORT_AFTER_PATH
TIMESFORMER_WINDOW_PREDICTIONS_PATH = FIG_DIR / "process2_timesformer_window_predictions.csv"

METHOD_ORDER = ["SORT", "LiveCellX", "Ultrack"]
METHOD_DISPLAY_LABELS = {
    "SORT": "SORT",
    "LiveCellX": "LiveCellX",
    "Ultrack": "Ultrack",
}
METHOD_COLORS = {
    "SORT": "#3F6DB5",
    "LiveCellX": "#F28E2B",
    "Ultrack": "#009E73",
}
METHOD_SHORT_LABELS = ["SORT", "LiveCellX", "Ultrack"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.bbox": "tight",
    }
)


@dataclass
class FrameCandidate:
    tid: int
    sc: SingleCellStatic
    bbox: np.ndarray


@dataclass
class MatchConfig:
    iou_threshold: float = 0.5
    min_candidate_iou: float = 0.05


def progress_iter(iterable: Iterable, total: Optional[int] = None, desc: str = ""):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        def _generator():
            for idx, item in enumerate(iterable, start=1):
                if idx == 1 or idx % 100 == 0 or (total is not None and idx == total):
                    print(f"[{desc}] {idx}/{total if total is not None else '?'}")
                yield item

        return _generator()


def resolve_path(path: Path, fallback: Optional[Path] = None) -> Path:
    if path.exists():
        return path
    if fallback is not None and fallback.exists():
        print(f"[warn] Missing {path}")
        print(f"[warn] Using fallback {fallback}")
        return fallback
    raise FileNotFoundError(path)


def load_sctc(path: Path) -> SingleCellTrajectoryCollection:
    print(f"[load] {path}")
    return SingleCellTrajectoryCollection.load_from_json_file(str(path), parallel=False)


def sc_center(sc: SingleCellStatic) -> np.ndarray:
    if sc.bbox is not None:
        bbox = np.asarray(sc.bbox, dtype=float)
        return np.asarray([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=float)
    try:
        return np.asarray(sc.get_center(), dtype=float)
    except Exception:
        return np.asarray([np.nan, np.nan], dtype=float)


def set_unique_relation_ids(traj: SingleCellTrajectory) -> None:
    traj.meta[SingleCellTrajectory.META_MOTHER_IDS] = sorted(
        {int(t.track_id) for t in traj.mother_trajectories}
    )
    traj.meta[SingleCellTrajectory.META_DAUGHTER_IDS] = sorted(
        {int(t.track_id) for t in traj.daughter_trajectories}
    )


def add_lineage_relation(
    mother: SingleCellTrajectory,
    daughter: Optional[SingleCellTrajectory],
) -> None:
    if daughter is None or int(mother.track_id) == int(daughter.track_id):
        return
    mother.add_daughter(daughter)
    daughter.add_mother(mother)
    set_unique_relation_ids(mother)
    set_unique_relation_ids(daughter)


def relation_id_set(traj: SingleCellTrajectory, attr_name: str, meta_key: str) -> Set[int]:
    ids = {int(t.track_id) for t in getattr(traj, attr_name, set()) or set()}
    meta = getattr(traj, "meta", {}) or {}
    ids.update(int(x) for x in meta.get(meta_key, []) or [] if x is not None)
    return ids


def has_existing_mother(traj: SingleCellTrajectory) -> bool:
    return bool(relation_id_set(traj, "mother_trajectories", SingleCellTrajectory.META_MOTHER_IDS))


def summarize_lineage_relation_consistency(sctc: SingleCellTrajectoryCollection) -> dict:
    mother_to_daughters = {}
    daughter_to_mothers = {}
    for tid, traj in sctc:
        tid = int(tid)
        daughter_ids = relation_id_set(traj, "daughter_trajectories", SingleCellTrajectory.META_DAUGHTER_IDS)
        mother_ids = relation_id_set(traj, "mother_trajectories", SingleCellTrajectory.META_MOTHER_IDS)
        if daughter_ids:
            mother_to_daughters[tid] = sorted(daughter_ids)
        if mother_ids:
            daughter_to_mothers[tid] = sorted(mother_ids)
    duplicate_daughters = {
        tid: mids for tid, mids in daughter_to_mothers.items() if len(mids) > 1
    }
    daughter_count_values = [len(v) for v in mother_to_daughters.values()]
    mother_count_values = [len(v) for v in daughter_to_mothers.values()]
    daughter_count_distribution = {
        int(count): int(daughter_count_values.count(count))
        for count in sorted(set(daughter_count_values))
    }
    mother_count_distribution = {
        int(count): int(mother_count_values.count(count))
        for count in sorted(set(mother_count_values))
    }
    return {
        "n_mother_tracks_with_daughters": len(mother_to_daughters),
        "n_mother_to_daughter_edges": int(sum(len(v) for v in mother_to_daughters.values())),
        "n_mothers_with_one_daughter": int(daughter_count_distribution.get(1, 0)),
        "n_mothers_with_two_daughters": int(daughter_count_distribution.get(2, 0)),
        "n_mothers_with_more_than_two_daughters": int(sum(1 for x in daughter_count_values if x > 2)),
        "daughters_per_mother_distribution": json.dumps(daughter_count_distribution, sort_keys=True),
        "n_daughter_tracks_with_mother": len(daughter_to_mothers),
        "n_daughters_with_one_mother": int(mother_count_distribution.get(1, 0)),
        "n_daughters_with_more_than_one_mother": int(sum(1 for x in mother_count_values if x > 1)),
        "mothers_per_daughter_distribution": json.dumps(mother_count_distribution, sort_keys=True),
        "n_duplicate_daughter_tracks": len(duplicate_daughters),
        "duplicate_daughter_to_mothers": duplicate_daughters,
    }


def write_timesformer_lineage_consistency(
    sctc: SingleCellTrajectoryCollection,
    fig_dir: Path,
) -> dict:
    consistency = summarize_lineage_relation_consistency(sctc)
    pd.DataFrame(
        [
            {
                key: value
                for key, value in consistency.items()
                if key != "duplicate_daughter_to_mothers"
            }
        ]
    ).to_csv(fig_dir / "process2_timesformer_lineage_consistency.csv", index=False)
    duplicate_df = pd.DataFrame(
        [
            {"daughter_track_id": tid, "mother_track_ids": ",".join(str(x) for x in mids)}
            for tid, mids in consistency["duplicate_daughter_to_mothers"].items()
        ]
    )
    duplicate_df.to_csv(fig_dir / "process2_timesformer_duplicate_daughters.csv", index=False)
    print(
        "[timesformer] lineage consistency: "
        f"{consistency['n_mother_tracks_with_daughters']} mothers, "
        f"{consistency['n_mother_to_daughter_edges']} mother-daughter edges, "
        f"{consistency['n_mothers_with_two_daughters']} mothers with exactly two daughters, "
        f"{consistency['n_duplicate_daughter_tracks']} duplicate daughter assignments"
    )
    return consistency


def configure_timesformer_test_pipeline(model, window_size: int) -> None:
    """Force the inference pipeline to consume the generated 8-frame clips.

    The training config samples frames with a larger interval from longer videos.
    Here each generated trajectory clip already has exactly the local sliding
    window, so inference must sample consecutive frames.
    """
    model.cfg.test_pipeline = [
        dict(io_backend="disk", type="DecordInit"),
        dict(clip_len=int(window_size), frame_interval=1, num_clips=1, test_mode=True, type="SampleFrames"),
        dict(type="DecordDecode"),
        dict(scale=(-1, 224), type="Resize"),
        dict(crop_size=224, type="ThreeCrop"),
        dict(input_format="NCTHW", type="FormatShape"),
        dict(type="PackActionInputs"),
    ]


def load_timesformer_model(config_path: Path, checkpoint_path: Path, device: str, window_size: int):
    try:
        from mmaction.apis import init_recognizer
    except Exception as exc:
        raise RuntimeError(
            "Could not import mmaction. Run the TimeSformer correction in the MMAction/MMEngine "
            "environment, or run without --run-timesformer-correction."
        ) from exc

    config_path = resolve_path(Path(config_path))
    checkpoint_path = resolve_path(Path(checkpoint_path))
    print(f"[timesformer] config: {config_path}")
    print(f"[timesformer] checkpoint: {checkpoint_path}")
    # PyTorch >= 2.6 defaults torch.load(weights_only=True), but MMEngine
    # checkpoints include training metadata classes. This checkpoint is local
    # and user-provided, so load it as a trusted full checkpoint.
    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"[timesformer] requested {device}, but CUDA is not available; falling back to CPU")
        device = "cpu"

    original_torch_load = torch.load

    def trusted_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_torch_load
    try:
        model = init_recognizer(str(config_path), str(checkpoint_path), device=device)
    finally:
        torch.load = original_torch_load
    configure_timesformer_test_pipeline(model, window_size=window_size)
    return model


def get_prediction_label_and_score(result, division_label: int) -> Tuple[int, float]:
    label = None
    score = np.nan
    if hasattr(result, "pred_label"):
        label_obj = result.pred_label
        label = int(label_obj.item() if hasattr(label_obj, "item") else label_obj)
    elif hasattr(result, "pred_labels"):
        label_obj = result.pred_labels
        label = int(label_obj.item() if hasattr(label_obj, "item") else label_obj)
    elif isinstance(result, dict):
        if "pred_label" in result:
            label = int(result["pred_label"])
        elif "pred_labels" in result:
            label = int(result["pred_labels"])
    if hasattr(result, "pred_score"):
        scores = result.pred_score
        try:
            score = float(scores[int(division_label)].detach().cpu().item())
        except Exception:
            try:
                score = float(scores[int(division_label)])
            except Exception:
                score = np.nan
    if label is None:
        raise ValueError(f"Could not read prediction label from MMAction result: {type(result)}")
    return label, score


def infer_timesformer_windows_for_traj(
    traj: SingleCellTrajectory,
    model,
    out_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    from mmaction.apis import inference_recognizer
    from livecellx.track.classify_utils import gen_inference_sctc_sample_videos

    if len(traj) < int(args.timesformer_window_size):
        return pd.DataFrame()

    tmp_sctc = SingleCellTrajectoryCollection([traj])
    sample_df = gen_inference_sctc_sample_videos(
        tmp_sctc,
        class_label="unknown",
        window_size=int(args.timesformer_window_size),
        step_size=int(args.timesformer_step_size),
        prefix=f"tid{int(traj.track_id)}",
        out_dir=str(out_dir),
        padding_pixels=[int(args.timesformer_padding)],
        fps=3,
    )
    sample_df = sample_df[sample_df["frame_type"].astype(str) == str(args.timesformer_frame_type)].copy()
    rows = []
    for row in progress_iter(
        sample_df.to_dict("records"),
        total=len(sample_df),
        desc=f"TimeSformer tid {int(traj.track_id)}",
    ):
        video_path = out_dir / "videos" / str(row["path"])
        try:
            result = inference_recognizer(model, str(video_path))
            pred_label, division_score = get_prediction_label_and_score(result, args.timesformer_division_label)
            error = ""
        except Exception as exc:
            pred_label = -1
            division_score = np.nan
            error = repr(exc)
        rows.append(
            {
                "track_id": int(traj.track_id),
                "start_time": int(row["start_time"]),
                "end_time": int(row["end_time"]),
                "video_path": str(video_path),
                "pred_label": int(pred_label),
                "division_score": float(division_score) if pd.notna(division_score) else np.nan,
                "is_division": bool(int(pred_label) == int(args.timesformer_division_label)),
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def choose_split_from_predictions(pred_df: pd.DataFrame, args: argparse.Namespace) -> Optional[Dict[str, object]]:
    """Choose split as first negative window start minus one after a positive run."""
    if pred_df.empty or "is_division" not in pred_df:
        return None
    pred_df = pred_df.sort_values("start_time").reset_index(drop=True)
    positives = pred_df[pred_df["is_division"].astype(bool)]
    if positives.empty:
        return None

    groups = []
    current = []
    prev_start = None
    max_gap = int(args.timesformer_max_gap)
    for row in positives.to_dict("records"):
        start = int(row["start_time"])
        if prev_start is None or start - prev_start <= int(args.timesformer_step_size) * (max_gap + 1):
            current.append(row)
        else:
            groups.append(current)
            current = [row]
        prev_start = start
    if current:
        groups.append(current)

    valid_groups = [g for g in groups if len(g) >= int(args.timesformer_min_positive_windows)]
    if not valid_groups:
        return None

    best_group = max(
        valid_groups,
        key=lambda g: (
            len(g),
            np.nanmean([float(x.get("division_score", np.nan)) for x in g]),
            int(g[-1]["start_time"]) - int(g[0]["start_time"]),
        ),
    )
    last_positive_start = int(best_group[-1]["start_time"])
    later = pred_df[pred_df["start_time"].astype(int) > last_positive_start].copy()
    later_negative = later[~later["is_division"].astype(bool)]
    if len(later_negative):
        first_negative_start = int(later_negative.iloc[0]["start_time"])
        split_t = first_negative_start - 1
        split_rule = "first_negative_after_positive_minus_one"
    else:
        split_t = int(best_group[-1]["end_time"])
        split_rule = "positive_run_end_no_later_negative"

    return {
        "split_t": int(split_t),
        "positive_start": int(best_group[0]["start_time"]),
        "positive_end": int(best_group[-1]["end_time"]),
        "n_positive_windows": int(len(best_group)),
        "split_rule": split_rule,
    }


def split_trajectory_after_time(
    sctc: SingleCellTrajectoryCollection,
    traj: SingleCellTrajectory,
    split_t: int,
    new_track_id: int,
) -> Optional[SingleCellTrajectory]:
    times = sorted(int(t) for t in traj.times)
    pre_times = [t for t in times if t <= int(split_t)]
    post_times = [t for t in times if t > int(split_t)]
    if not pre_times or not post_times:
        return None
    post_cells = {int(t): traj.timeframe_to_single_cell[int(t)] for t in post_times}
    for t in post_times:
        traj.timeframe_to_single_cell.pop(int(t), None)

    daughter = SingleCellTrajectory(
        track_id=int(new_track_id),
        timeframe_to_single_cell=post_cells,
        img_dataset=traj.img_dataset,
        mask_dataset=traj.mask_dataset,
        extra_datasets=traj.extra_datasets,
        mother_trajectories=set(),
        daughter_trajectories=set(),
        meta={
            SingleCellTrajectory.META_MOTHER_IDS: [],
            SingleCellTrajectory.META_DAUGHTER_IDS: [],
        },
    )
    sctc.add_trajectory(daughter)
    return daughter


def find_second_daughter(
    sctc: SingleCellTrajectoryCollection,
    mother: SingleCellTrajectory,
    daughter_a: SingleCellTrajectory,
    split_t: int,
    search_window: int,
    max_distance: float,
) -> Tuple[Optional[SingleCellTrajectory], float, Optional[int]]:
    if not mother.times:
        return None, np.nan, None
    mother_time = max(int(t) for t in mother.times if int(t) <= int(split_t))
    mother_center = sc_center(mother.get_sc(mother_time))
    if not np.isfinite(mother_center).all():
        return None, np.nan, None
    daughter_a_times = set(int(t) for t in daughter_a.times)

    best = None
    best_score = np.inf
    best_time = None
    for cand_id, cand in sctc:
        cand_id = int(cand_id)
        if cand_id in {int(mother.track_id), int(daughter_a.track_id)}:
            continue
        if has_existing_mother(cand):
            continue
        cand_times = sorted(int(t) for t in cand.times)
        if not cand_times:
            continue
        window_times = [
            t for t in cand_times
            if int(split_t) + 1 <= int(t) <= int(split_t) + int(search_window)
        ]
        if not window_times:
            continue
        for t in window_times:
            cand_center = sc_center(cand.get_sc(t))
            if not np.isfinite(cand_center).all():
                continue
            mother_dist = float(np.linalg.norm(cand_center - mother_center))
            if mother_dist > float(max_distance):
                continue
            daughter_penalty = 0.0
            if t in daughter_a_times:
                daughter_penalty = 0.25 * float(np.linalg.norm(cand_center - sc_center(daughter_a.get_sc(t))))
            time_penalty = 2.0 * float(t - (int(split_t) + 1))
            score = mother_dist + daughter_penalty + time_penalty
            if score < best_score:
                best = cand
                best_score = score
                best_time = int(t)
    return best, float(best_score) if np.isfinite(best_score) else np.nan, best_time


def select_timesformer_candidate_tids(
    target_sctc: SingleCellTrajectoryCollection,
    gt_sctc: SingleCellTrajectoryCollection,
    edges_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Set[int]:
    """Select after-CSNet target tracks that overlap exact GT division anchors.

    This only narrows the expensive model-inference set.  It does not provide
    the split frame or the daughter identities used by TimeSformer correction.
    """
    exact_anchor_df = build_division_anchor_cell_table(gt_sctc, edges_df, tolerance_max=0)
    if exact_anchor_df.empty:
        return set()
    match_df = match_sctc_condition(
        "TimeSformer candidate selection",
        target_sctc,
        gt_sctc,
        exact_anchor_df,
        MatchConfig(iou_threshold=float(args.iou_threshold)),
    )
    tids = set(int(x) for x in match_df.loc[match_df["matched"], "target_tid"].dropna().tolist())
    print(f"[timesformer] selected {len(tids)} after-CSNet trajectories from exact GT anchors")
    return tids


def apply_timesformer_lineage_correction(
    sctc: SingleCellTrajectoryCollection,
    candidate_tids: Optional[Set[int]],
    args: argparse.Namespace,
    fig_dir: Path,
) -> Tuple[SingleCellTrajectoryCollection, pd.DataFrame, pd.DataFrame]:
    model = load_timesformer_model(
        Path(args.timesformer_config_path),
        Path(args.timesformer_model_path),
        args.timesformer_device,
        int(args.timesformer_window_size),
    )
    video_dir = Path(args.timesformer_video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    if args.timesformer_all_trajectories or candidate_tids is None:
        process_ids = sorted(int(x) for x in sctc.get_track_ids())
    else:
        process_ids = sorted(int(x) for x in candidate_tids if int(x) in sctc.track_id_to_trajectory)
    if int(args.timesformer_max_trajectories) > 0:
        process_ids = process_ids[: int(args.timesformer_max_trajectories)]
    print(f"[timesformer] running on {len(process_ids)} trajectories")

    correction_rows = []
    prediction_rows = []
    next_tid = int(sctc.get_max_tid()) + 1
    for tid in progress_iter(process_ids, total=len(process_ids), desc="TimeSformer correction"):
        if int(tid) not in sctc.track_id_to_trajectory:
            continue
        traj = sctc.get_trajectory(int(tid))
        pred_df = infer_timesformer_windows_for_traj(traj, model, video_dir, args)
        if len(pred_df):
            prediction_rows.extend(pred_df.to_dict("records"))
        split_info = choose_split_from_predictions(pred_df, args)
        if split_info is None:
            correction_rows.append(
                {
                    "source_track_id": int(tid),
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
            correction_rows.append(
                {
                    "source_track_id": int(tid),
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
        correction_rows.append(
            {
                "source_track_id": int(tid),
                "status": "corrected_with_two_daughters" if daughter_b is not None else "corrected_one_daughter_only",
                **split_info,
                "daughter_a_track_id": int(daughter_a.track_id),
                "daughter_b_track_id": int(daughter_b.track_id) if daughter_b is not None else np.nan,
                "daughter_b_score": daughter_b_score,
                "daughter_b_match_time": daughter_b_time,
            }
        )

    correction_df = pd.DataFrame(correction_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    correction_df.to_csv(fig_dir / "process2_timesformer_lineage_corrections.csv", index=False)
    prediction_df.to_csv(fig_dir / "process2_timesformer_window_predictions.csv", index=False)
    write_timesformer_lineage_consistency(sctc, fig_dir)
    if args.save_timesformer_corrected_sctc:
        output_path = Path(args.timesformer_output_sctc)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sctc.write_json(str(output_path))
        print(f"[timesformer] wrote corrected after-CSNet SCTC: {output_path}")
    else:
        print("[timesformer] skipped writing full corrected SCTC JSON; use --save-timesformer-corrected-sctc if needed")
    return sctc, correction_df, prediction_df


def apply_timesformer_lineage_correction_from_predictions(
    sctc: SingleCellTrajectoryCollection,
    prediction_csv: Path,
    args: argparse.Namespace,
    fig_dir: Path,
) -> Tuple[SingleCellTrajectoryCollection, pd.DataFrame, pd.DataFrame]:
    """Replay lineage correction from saved TimeSformer window predictions.

    This avoids rerunning the GPU model when only the correction rule changes.
    It starts from the original after-CSNet SCTC and applies the current split
    and daughter-selection logic to the already-saved per-window predictions.
    """
    prediction_csv = resolve_path(Path(prediction_csv))
    prediction_df = pd.read_csv(prediction_csv)
    required = {"track_id", "start_time", "end_time", "pred_label", "division_score", "is_division"}
    missing = sorted(required - set(prediction_df.columns))
    if missing:
        raise ValueError(f"Saved TimeSformer prediction CSV is missing columns: {missing}")
    if prediction_df.empty:
        raise ValueError(f"Saved TimeSformer prediction CSV is empty: {prediction_csv}")

    prediction_df = prediction_df.copy()
    prediction_df["track_id"] = prediction_df["track_id"].astype(int)
    process_ids = sorted(
        int(x)
        for x in prediction_df["track_id"].dropna().unique()
        if int(x) in sctc.track_id_to_trajectory
    )
    if int(args.timesformer_max_trajectories) > 0:
        process_ids = process_ids[: int(args.timesformer_max_trajectories)]
    print(
        f"[timesformer] replaying saved predictions for {len(process_ids)} trajectories "
        f"from {prediction_csv}"
    )

    correction_rows = []
    next_tid = int(sctc.get_max_tid()) + 1
    for tid in progress_iter(process_ids, total=len(process_ids), desc="Replay TimeSformer correction"):
        if int(tid) not in sctc.track_id_to_trajectory:
            continue
        traj = sctc.get_trajectory(int(tid))
        pred_df = prediction_df[prediction_df["track_id"].astype(int) == int(tid)].copy()
        split_info = choose_split_from_predictions(pred_df, args)
        if split_info is None:
            correction_rows.append(
                {
                    "source_track_id": int(tid),
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
            correction_rows.append(
                {
                    "source_track_id": int(tid),
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
        correction_rows.append(
            {
                "source_track_id": int(tid),
                "status": "corrected_with_two_daughters" if daughter_b is not None else "corrected_one_daughter_only",
                **split_info,
                "daughter_a_track_id": int(daughter_a.track_id),
                "daughter_b_track_id": int(daughter_b.track_id) if daughter_b is not None else np.nan,
                "daughter_b_score": daughter_b_score,
                "daughter_b_match_time": daughter_b_time,
            }
        )

    correction_df = pd.DataFrame(correction_rows)
    correction_df.to_csv(fig_dir / "process2_timesformer_lineage_corrections.csv", index=False)
    write_timesformer_lineage_consistency(sctc, fig_dir)
    if args.save_timesformer_corrected_sctc:
        output_path = Path(args.timesformer_output_sctc)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sctc.write_json(str(output_path))
        print(f"[timesformer] wrote replay-corrected after-CSNet SCTC: {output_path}")
    else:
        print("[timesformer] skipped writing full corrected SCTC JSON; use --save-timesformer-corrected-sctc if needed")
    return sctc, correction_df, prediction_df


def relation_ids_from_gt(gt_sctc: SingleCellTrajectoryCollection) -> Set[int]:
    """Return GT trajectories involved in any mother/daughter relation."""
    relation_ids: Set[int] = set()
    for tid, traj in gt_sctc:
        mother_ids = {int(t.track_id) for t in traj.mother_trajectories}
        daughter_ids = {int(t.track_id) for t in traj.daughter_trajectories}
        meta = traj.meta or {}
        mother_ids.update(int(x) for x in meta.get("mother_trajectory_ids", []) if x is not None)
        daughter_ids.update(int(x) for x in meta.get("daughter_trajectory_ids", []) if x is not None)
        if mother_ids or daughter_ids:
            relation_ids.add(int(tid))
            relation_ids.update(mother_ids)
            relation_ids.update(daughter_ids)
    existing = set(int(x) for x in gt_sctc.track_id_to_trajectory.keys())
    return relation_ids.intersection(existing)


def lineage_edges_from_gt(
    gt_sctc: SingleCellTrajectoryCollection,
    gt_ids: Set[int],
    min_frame: int = 0,
    max_frame: Optional[int] = 99,
) -> pd.DataFrame:
    rows = []
    for tid in sorted(gt_ids):
        traj = gt_sctc.get_trajectory(tid)
        daughter_ids = {int(t.track_id) for t in traj.daughter_trajectories}
        daughter_ids.update(int(x) for x in (traj.meta or {}).get("daughter_trajectory_ids", []) if x is not None)
        mother_times = [int(t) for t in traj.times if frame_in_eval_window(int(t), min_frame, max_frame)]
        if not mother_times:
            continue
        for daughter_id in sorted(d for d in daughter_ids if d in gt_ids):
            daughter = gt_sctc.get_trajectory(daughter_id)
            daughter_times = [int(t) for t in daughter.times if frame_in_eval_window(int(t), min_frame, max_frame)]
            if not daughter_times:
                continue
            mother_end = int(max(mother_times))
            daughter_start = int(min(daughter_times))
            if not (frame_in_eval_window(mother_end, min_frame, max_frame) and frame_in_eval_window(daughter_start, min_frame, max_frame)):
                continue
            rows.append(
                {
                    "mother_gt_tid": int(tid),
                    "daughter_gt_tid": int(daughter_id),
                    "mother_start": int(min(mother_times)),
                    "mother_end": mother_end,
                    "daughter_start": daughter_start,
                    "daughter_end": int(max(daughter_times)),
                    "division_gap": int(daughter_start - mother_end),
                }
            )
    return pd.DataFrame(rows)


def validate_gt_scope(
    gt_ids: Set[int],
    edges_df: pd.DataFrame,
    full_gt_cell_df: pd.DataFrame,
    eval_gt_cell_df: pd.DataFrame,
) -> None:
    """Fail early if the benchmark is not anchored to annotated GT cells."""
    if not gt_ids:
        raise ValueError("No GT trajectories with mother/daughter relations were found.")
    if edges_df.empty:
        raise ValueError("No GT mother-daughter edges were found.")
    if full_gt_cell_df.empty:
        raise ValueError("No cells were found in the selected GT trajectories.")
    if eval_gt_cell_df.empty:
        raise ValueError("No GT cells were selected for evaluation.")

    full_ids = set(int(x) for x in full_gt_cell_df["gt_tid"].unique())
    eval_ids = set(int(x) for x in eval_gt_cell_df["gt_tid"].unique())
    missing_from_gt = (full_ids | eval_ids) - set(int(x) for x in gt_ids)
    if missing_from_gt:
        raise ValueError(f"Evaluation contains non-GT trajectory IDs: {sorted(missing_from_gt)[:10]}")


def write_benchmark_scope(
    fig_dir: Path,
    args: argparse.Namespace,
    gt_ids: Set[int],
    edges_df: pd.DataFrame,
    full_gt_cell_df: pd.DataFrame,
    eval_gt_cell_df: pd.DataFrame,
    scope: str,
    branch_gt_cell_df: Optional[pd.DataFrame] = None,
) -> None:
    rows = [
        ("gt_path", str(resolve_path(Path(args.gt_path)))),
        ("sort_before_path", str(resolve_path(Path(args.sort_before_path), Path(args.sort_before_fallback_path)))),
        ("sort_after_path", str(resolve_path(Path(args.sort_after_path)))),
        (
            "timesformer_lineage_correction_enabled",
            str(bool(args.run_timesformer_correction or args.replay_timesformer_predictions or args.use_existing_timesformer_correction)),
        ),
        (
            "timesformer_lineage_correction_mode",
            (
                "fresh_model_inference"
                if args.run_timesformer_correction
                else "replay_saved_predictions"
                if args.replay_timesformer_predictions
                else "existing_corrected_sctc"
                if args.use_existing_timesformer_correction
                else "disabled"
            ),
        ),
        ("timesformer_model_path", str(Path(args.timesformer_model_path))),
        ("timesformer_config_path", str(Path(args.timesformer_config_path))),
        ("timesformer_prediction_csv", str(Path(args.timesformer_prediction_csv))),
        ("timesformer_corrected_after_sctc_path", str(Path(args.timesformer_output_sctc))),
        ("ultrack_zarr_path", str(Path(args.ultrack_zarr_path))),
        ("evaluation_scope", scope),
        ("iou_threshold", str(args.iou_threshold)),
        ("min_frame", str(args.min_frame)),
        ("max_frame", str(args.max_frame)),
        ("branch_tolerances", " ".join(str(x) for x in args.branch_tolerances)),
        ("n_gt_relation_trajectories", str(len(gt_ids))),
        ("n_gt_mother_daughter_edges", str(len(edges_df))),
        ("n_full_gt_cells_in_relation_trajectories", str(len(full_gt_cell_df))),
        ("n_evaluation_gt_cells", str(len(eval_gt_cell_df))),
        ("n_branch_anchor_tolerance_gt_cells", str(0 if branch_gt_cell_df is None else len(branch_gt_cell_df))),
    ]
    pd.DataFrame(rows, columns=["field", "value"]).to_csv(
        fig_dir / "process2_ctc_style_benchmark_scope.csv",
        index=False,
    )


def bbox_intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[0] >= b[2] or a[2] <= b[0] or a[1] >= b[3] or a[3] <= b[1])


def bbox_intersection(a: Sequence[float], b: Sequence[float]) -> Optional[np.ndarray]:
    bbox = np.asarray(
        [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])],
        dtype=int,
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def bbox_area(bbox: Sequence[float]) -> float:
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def cached_mask_area(sc: SingleCellStatic, area_cache: Dict[int, float]) -> float:
    key = id(sc)
    if key not in area_cache:
        try:
            area_cache[key] = float(sc.get_contour_mask(bbox=sc.bbox).astype(bool).sum())
        except Exception:
            area_cache[key] = np.nan
    return area_cache[key]


def cropped_iou(
    gt_sc: SingleCellStatic,
    cand_sc: SingleCellStatic,
    gt_area: float,
    cand_area: float,
) -> float:
    inter_bbox = bbox_intersection(gt_sc.bbox, cand_sc.bbox)
    if inter_bbox is None or gt_area <= 0 or cand_area <= 0:
        return 0.0
    try:
        gt_mask = gt_sc.get_contour_mask(bbox=inter_bbox).astype(bool)
        cand_mask = cand_sc.get_contour_mask(bbox=inter_bbox).astype(bool)
        inter = float(np.logical_and(gt_mask, cand_mask).sum())
    except Exception:
        try:
            return float(gt_sc.compute_iou(cand_sc))
        except Exception:
            return 0.0
    union = gt_area + cand_area - inter
    return float(inter / union) if union > 0 else 0.0


def frame_in_eval_window(t: int, min_frame: int = 0, max_frame: Optional[int] = 99) -> bool:
    t = int(t)
    if min_frame is not None and t < int(min_frame):
        return False
    if max_frame is not None and t > int(max_frame):
        return False
    return True


def build_gt_cell_table(
    gt_sctc: SingleCellTrajectoryCollection,
    gt_ids: Set[int],
    min_frame: int = 0,
    max_frame: Optional[int] = 99,
) -> pd.DataFrame:
    rows = []
    for gt_tid in sorted(gt_ids):
        traj = gt_sctc.get_trajectory(gt_tid)
        eval_times = [int(t) for t in sorted(int(x) for x in traj.times) if frame_in_eval_window(int(t), min_frame, max_frame)]
        if not eval_times:
            continue
        for t in eval_times:
            sc = traj.get_sc(t)
            if sc.bbox is None:
                continue
            rows.append(
                {
                    "gt_tid": int(gt_tid),
                    "time": int(t),
                    "gt_start": int(min(eval_times)),
                    "gt_end": int(max(eval_times)),
                    "gt_length": int(len(eval_times)),
                }
            )
    return pd.DataFrame(rows)


def build_division_anchor_cell_table(
    gt_sctc: SingleCellTrajectoryCollection,
    edges_df: pd.DataFrame,
    tolerance_max: int,
    min_frame: int = 0,
    max_frame: Optional[int] = 99,
) -> pd.DataFrame:
    """Build mother-last/daughter-start anchor rows plus tolerance windows.

    Exact anchors have offset 0.  Extra rows exist only so BC(i) can be computed
    with CTC-style frame tolerance without matching every cell in every GT
    trajectory.
    """
    rows = []
    for mother_id, group in edges_df.groupby("mother_gt_tid"):
        daughters = sorted(int(x) for x in group["daughter_gt_tid"].tolist())[:2]
        if len(daughters) < 2:
            continue
        mother_traj = gt_sctc.get_trajectory(int(mother_id))
        mother_times = set(int(x) for x in mother_traj.times if frame_in_eval_window(int(x), min_frame, max_frame))
        if not mother_times:
            continue
        mother_end = int(max(mother_times))
        daughter_starts = {
            int(row["daughter_gt_tid"]): int(row["daughter_start"])
            for _, row in group.iterrows()
        }

        specs = [
            {
                "event_id": int(mother_id),
                "role": "mother",
                "gt_tid": int(mother_id),
                "anchor_time": mother_end,
                "candidate_times": range(mother_end - tolerance_max, mother_end + 1),
            },
            {
                "event_id": int(mother_id),
                "role": "daughter1",
                "gt_tid": daughters[0],
                "anchor_time": int(daughter_starts[daughters[0]]),
                "candidate_times": range(int(daughter_starts[daughters[0]]), int(daughter_starts[daughters[0]]) + tolerance_max + 1),
            },
            {
                "event_id": int(mother_id),
                "role": "daughter2",
                "gt_tid": daughters[1],
                "anchor_time": int(daughter_starts[daughters[1]]),
                "candidate_times": range(int(daughter_starts[daughters[1]]), int(daughter_starts[daughters[1]]) + tolerance_max + 1),
            },
        ]

        for spec in specs:
            traj = gt_sctc.get_trajectory(int(spec["gt_tid"]))
            traj_times = set(int(x) for x in traj.times if frame_in_eval_window(int(x), min_frame, max_frame))
            for t in spec["candidate_times"]:
                t = int(t)
                if not frame_in_eval_window(t, min_frame, max_frame):
                    continue
                if t not in traj_times:
                    continue
                rows.append(
                    {
                        "event_id": spec["event_id"],
                        "role": spec["role"],
                        "gt_tid": int(spec["gt_tid"]),
                        "time": t,
                        "anchor_time": int(spec["anchor_time"]),
                        "offset": abs(t - int(spec["anchor_time"])),
                        "gt_start": int(min(traj_times)),
                        "gt_end": int(max(traj_times)),
                        "gt_length": int(len(traj_times)),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(["event_id", "role", "gt_tid", "time"]).reset_index(drop=True)


def exact_anchor_mapping(anchor_match_df: pd.DataFrame) -> pd.DataFrame:
    """Return one exact anchor row per GT trajectory/role/event."""
    exact = anchor_match_df[anchor_match_df["offset"].astype(int) == 0].copy()
    cols = [
        "condition",
        "event_id",
        "role",
        "gt_tid",
        "time",
        "target_tid",
        "iou",
        "matched",
        "gt_start",
        "gt_end",
        "gt_length",
    ]
    return exact[cols].sort_values(["condition", "event_id", "role"]).reset_index(drop=True)


def attach_branch_anchor_matches(
    condition: str,
    full_match_df: pd.DataFrame,
    branch_gt_cell_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach full one-to-one match results to division-anchor tolerance rows."""

    cols = ["gt_tid", "time", "target_tid", "iou", "matched"]
    cond_match = full_match_df[full_match_df["condition"] == condition][cols].copy()
    out = branch_gt_cell_df.merge(cond_match, on=["gt_tid", "time"], how="left")
    out["condition"] = condition
    out["target_tid"] = out["target_tid"].where(out["matched"].fillna(False), None)
    out["iou"] = out["iou"].fillna(0.0)
    out["matched"] = out["matched"].fillna(False).astype(bool)
    return out


def build_time_index(
    sctc: SingleCellTrajectoryCollection,
    allowed_times: Set[int],
) -> Dict[int, List[FrameCandidate]]:
    index: Dict[int, List[FrameCandidate]] = {int(t): [] for t in allowed_times}
    for tid, traj in sctc:
        for t in traj.timeframe_set.intersection(allowed_times):
            sc = traj.get_sc(int(t))
            if sc.bbox is None:
                continue
            index[int(t)].append(FrameCandidate(int(tid), sc, np.asarray(sc.bbox, dtype=int)))
    return index


def best_sctc_match(
    gt_sc: SingleCellStatic,
    candidates: List[FrameCandidate],
    cfg: MatchConfig,
    area_cache: Dict[int, float],
) -> Tuple[Optional[int], float]:
    if gt_sc.bbox is None:
        return None, 0.0
    gt_area = cached_mask_area(gt_sc, area_cache)
    if not np.isfinite(gt_area) or gt_area <= 0:
        return None, 0.0
    best_tid = None
    best_iou = 0.0
    for cand in candidates:
        if not bbox_intersects(gt_sc.bbox, cand.bbox):
            continue
        cand_area = cached_mask_area(cand.sc, area_cache)
        if not np.isfinite(cand_area) or cand_area <= 0:
            continue
        inter_bbox = bbox_intersection(gt_sc.bbox, cand.bbox)
        if inter_bbox is None:
            continue
        if bbox_area(inter_bbox) / max(gt_area, cand_area) < cfg.min_candidate_iou:
            continue
        iou = cropped_iou(gt_sc, cand.sc, gt_area, cand_area)
        if iou > best_iou:
            best_iou = iou
            best_tid = cand.tid
    return best_tid, best_iou


def one_to_one_sctc_matches_for_time(
    rows: List[dict],
    target_candidates: List[FrameCandidate],
    gt_sctc: SingleCellTrajectoryCollection,
    cfg: MatchConfig,
    area_cache: Dict[int, float],
) -> List[dict]:
    """Greedy one-to-one GT-to-method matching for one frame.

    This implements the partial-GT analogue of CTC object matching: each method
    cell can be assigned to at most one GT cell in the same frame.  Rows that do
    not receive an IoU >= threshold match are stored as unmatched with IoU = 0.
    """

    pairs: List[Tuple[float, int, int]] = []
    for gt_idx, row in enumerate(rows):
        gt_sc = gt_sctc.get_trajectory(int(row["gt_tid"])).get_sc(int(row["time"]))
        if gt_sc.bbox is None:
            continue
        gt_area = cached_mask_area(gt_sc, area_cache)
        if not np.isfinite(gt_area) or gt_area <= 0:
            continue
        for cand_idx, cand in enumerate(target_candidates):
            if not bbox_intersects(gt_sc.bbox, cand.bbox):
                continue
            cand_area = cached_mask_area(cand.sc, area_cache)
            if not np.isfinite(cand_area) or cand_area <= 0:
                continue
            inter_bbox = bbox_intersection(gt_sc.bbox, cand.bbox)
            if inter_bbox is None:
                continue
            if bbox_area(inter_bbox) / max(gt_area, cand_area) < cfg.min_candidate_iou:
                continue
            iou = cropped_iou(gt_sc, cand.sc, gt_area, cand_area)
            if iou > 0:
                pairs.append((float(iou), gt_idx, cand_idx))

    pairs.sort(key=lambda x: x[0], reverse=True)
    assigned_gt: Dict[int, Tuple[int, float]] = {}
    used_candidates: Set[int] = set()
    for iou, gt_idx, cand_idx in pairs:
        if gt_idx in assigned_gt or cand_idx in used_candidates:
            continue
        if iou < cfg.iou_threshold:
            continue
        assigned_gt[gt_idx] = (cand_idx, iou)
        used_candidates.add(cand_idx)

    out_rows = []
    for gt_idx, row in enumerate(rows):
        if gt_idx in assigned_gt:
            cand_idx, iou = assigned_gt[gt_idx]
            target_tid = int(target_candidates[cand_idx].tid)
            matched = True
        else:
            target_tid = None
            iou = 0.0
            matched = False
        out_rows.append(
            {
                **row,
                "target_tid": target_tid,
                "iou": float(iou),
                "matched": bool(matched),
            }
        )
    return out_rows


def match_sctc_condition(
    condition: str,
    target_sctc: SingleCellTrajectoryCollection,
    gt_sctc: SingleCellTrajectoryCollection,
    gt_cell_df: pd.DataFrame,
    cfg: MatchConfig,
) -> pd.DataFrame:
    allowed_times = set(int(x) for x in gt_cell_df["time"].unique())
    target_by_time = build_time_index(target_sctc, allowed_times)
    area_cache: Dict[int, float] = {}
    rows = []
    records = gt_cell_df.to_dict("records")
    print(f"[match] {condition}: {len(records)} GT cells over {len(allowed_times)} frames")
    for t, group in progress_iter(list(gt_cell_df.groupby("time", sort=True)), total=len(allowed_times), desc=f"match {condition}"):
        frame_rows = [{**row, "condition": condition} for row in group.to_dict("records")]
        rows.extend(
            one_to_one_sctc_matches_for_time(
                frame_rows,
                target_by_time.get(int(t), []),
                gt_sctc,
                cfg,
                area_cache,
            )
        )
    return pd.DataFrame(rows)


class ZarrV3Labels:
    def __init__(self, zarr_path: Path):
        self.zarr_path = zarr_path
        with open(zarr_path / "zarr.json") as f:
            meta = json.load(f)
        self.shape = tuple(int(x) for x in meta["shape"])
        self.dtype = np.dtype(meta["data_type"])
        self._frame_cache: Dict[int, np.ndarray] = {}
        self._area_cache: Dict[int, Dict[int, int]] = {}

    def read_frame(self, t: int) -> np.ndarray:
        if t in self._frame_cache:
            return self._frame_cache[t]
        if t < 0 or t >= self.shape[0]:
            raise IndexError(t)
        import zstandard

        chunk_path = self.zarr_path / "c" / str(t) / "0" / "0"
        with open(chunk_path, "rb") as f:
            raw = zstandard.ZstdDecompressor().decompress(f.read())
        frame = np.frombuffer(raw, dtype=self.dtype).reshape(self.shape[1:])
        self._frame_cache[t] = frame
        if len(self._frame_cache) > 12:
            oldest = next(iter(self._frame_cache))
            self._frame_cache.pop(oldest, None)
        return frame

    def label_areas(self, t: int) -> Dict[int, int]:
        if t not in self._area_cache:
            frame = self.read_frame(t)
            labels, counts = np.unique(frame, return_counts=True)
            self._area_cache[t] = {int(label): int(count) for label, count in zip(labels, counts) if int(label) > 0}
            if len(self._area_cache) > 12:
                oldest = next(iter(self._area_cache))
                self._area_cache.pop(oldest, None)
        return self._area_cache[t]


def best_zarr_label_match(
    gt_sc: SingleCellStatic,
    labels: ZarrV3Labels,
    t: int,
    area_cache: Dict[int, float],
) -> Tuple[Optional[int], float]:
    if gt_sc.bbox is None or t < 0 or t >= labels.shape[0]:
        return None, 0.0
    bbox = np.asarray(gt_sc.bbox, dtype=int)
    y0 = max(0, int(bbox[0]))
    x0 = max(0, int(bbox[1]))
    y1 = min(labels.shape[1], int(bbox[2]))
    x1 = min(labels.shape[2], int(bbox[3]))
    if y1 <= y0 or x1 <= x0:
        return None, 0.0
    clipped_bbox = np.asarray([y0, x0, y1, x1], dtype=int)
    gt_area = cached_mask_area(gt_sc, area_cache)
    if not np.isfinite(gt_area) or gt_area <= 0:
        return None, 0.0
    try:
        gt_mask = gt_sc.get_contour_mask(bbox=clipped_bbox).astype(bool)
    except Exception:
        return None, 0.0
    frame = labels.read_frame(t)
    crop = frame[y0:y1, x0:x1]
    inside = crop[gt_mask]
    inside = inside[inside > 0]
    if len(inside) == 0:
        return None, 0.0
    cand_labels, intersections = np.unique(inside, return_counts=True)
    areas = labels.label_areas(t)
    best_label = None
    best_iou = 0.0
    for label_value, inter in zip(cand_labels, intersections):
        pred_area = areas.get(int(label_value), 0)
        union = gt_area + pred_area - float(inter)
        iou = float(inter / union) if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_label = int(label_value)
    return best_label, best_iou


def one_to_one_zarr_matches_for_time(
    rows: List[dict],
    labels: ZarrV3Labels,
    gt_sctc: SingleCellTrajectoryCollection,
    t: int,
    cfg: MatchConfig,
    area_cache: Dict[int, float],
) -> List[dict]:
    """Greedy one-to-one GT-to-label matching for one zarr frame."""

    if t < 0 or t >= labels.shape[0]:
        return [{**row, "target_tid": None, "iou": 0.0, "matched": False} for row in rows]
    frame = labels.read_frame(t)
    label_areas = labels.label_areas(t)
    pair_rows: List[Tuple[float, int, int]] = []
    for gt_idx, row in enumerate(rows):
        gt_sc = gt_sctc.get_trajectory(int(row["gt_tid"])).get_sc(int(row["time"]))
        if gt_sc.bbox is None:
            continue
        bbox = np.asarray(gt_sc.bbox, dtype=int)
        y0 = max(0, int(bbox[0]))
        x0 = max(0, int(bbox[1]))
        y1 = min(labels.shape[1], int(bbox[2]))
        x1 = min(labels.shape[2], int(bbox[3]))
        if y1 <= y0 or x1 <= x0:
            continue
        clipped_bbox = np.asarray([y0, x0, y1, x1], dtype=int)
        gt_area = cached_mask_area(gt_sc, area_cache)
        if not np.isfinite(gt_area) or gt_area <= 0:
            continue
        try:
            gt_mask = gt_sc.get_contour_mask(bbox=clipped_bbox).astype(bool)
        except Exception:
            continue
        crop = frame[y0:y1, x0:x1]
        inside = crop[gt_mask]
        inside = inside[inside > 0]
        if len(inside) == 0:
            continue
        cand_labels, intersections = np.unique(inside, return_counts=True)
        for label_value, inter in zip(cand_labels, intersections):
            label_value = int(label_value)
            pred_area = label_areas.get(label_value, 0)
            union = gt_area + pred_area - float(inter)
            iou = float(inter / union) if union > 0 else 0.0
            if iou > 0:
                pair_rows.append((iou, gt_idx, label_value))

    pair_rows.sort(key=lambda x: x[0], reverse=True)
    assigned_gt: Dict[int, Tuple[int, float]] = {}
    used_labels: Set[int] = set()
    for iou, gt_idx, label_value in pair_rows:
        if gt_idx in assigned_gt or label_value in used_labels:
            continue
        if iou < cfg.iou_threshold:
            continue
        assigned_gt[gt_idx] = (label_value, iou)
        used_labels.add(label_value)

    out_rows = []
    for gt_idx, row in enumerate(rows):
        if gt_idx in assigned_gt:
            target_tid, iou = assigned_gt[gt_idx]
            matched = True
        else:
            target_tid = None
            iou = 0.0
            matched = False
        out_rows.append({**row, "target_tid": target_tid, "iou": float(iou), "matched": bool(matched)})
    return out_rows


def match_ultrack_condition(
    condition: str,
    zarr_path: Path,
    gt_sctc: SingleCellTrajectoryCollection,
    gt_cell_df: pd.DataFrame,
    cfg: MatchConfig,
) -> pd.DataFrame:
    labels = ZarrV3Labels(zarr_path)
    area_cache: Dict[int, float] = {}
    rows = []
    records = gt_cell_df.to_dict("records")
    print(f"[match] {condition}: {len(records)} GT cells against {zarr_path}")
    grouped = list(gt_cell_df.groupby("time", sort=True))
    for t, group in progress_iter(grouped, total=len(grouped), desc=f"match {condition}"):
        frame_rows = [{**row, "condition": condition} for row in group.to_dict("records")]
        rows.extend(one_to_one_zarr_matches_for_time(frame_rows, labels, gt_sctc, int(t), cfg, area_cache))
    return pd.DataFrame(rows)


def sctc_track_frames(
    sctc: SingleCellTrajectoryCollection,
    target_tids: Set[int],
) -> Dict[int, Set[int]]:
    frames: Dict[int, Set[int]] = {}
    for tid in target_tids:
        if tid in sctc.track_id_to_trajectory:
            frames[int(tid)] = set(int(x) for x in sctc.get_trajectory(int(tid)).times)
        else:
            frames[int(tid)] = set()
    return frames


def ultrack_track_frames(zarr_path: Path, target_tids: Set[int]) -> Dict[int, Set[int]]:
    labels = ZarrV3Labels(zarr_path)
    target_tids = {int(x) for x in target_tids if pd.notna(x)}
    frames: Dict[int, Set[int]] = {tid: set() for tid in target_tids}
    if not target_tids:
        return frames
    remaining = set(target_tids)
    print(f"[ultrack] scanning zarr frame presence for {len(target_tids)} anchor-selected tracks")
    for t in progress_iter(range(labels.shape[0]), total=labels.shape[0], desc="ultrack track frames"):
        frame_labels = set(int(x) for x in np.unique(labels.read_frame(int(t))) if int(x) > 0)
        present = frame_labels.intersection(target_tids)
        for tid in present:
            frames[tid].add(int(t))
        remaining.difference_update(present)
    return frames


def is_sctc_json_path(path: Path) -> bool:
    return Path(path).suffix.lower() == ".json"


def longest_continuous_overlap_fraction(gt_times: Sequence[int], target_frames: Set[int]) -> float:
    ordered = sorted(int(x) for x in gt_times)
    if not ordered:
        return np.nan
    best = 0
    cur = 0
    prev_time = None
    for t in ordered:
        if t in target_frames:
            if prev_time is not None and t == prev_time + 1:
                cur += 1
            else:
                cur = 1
            best = max(best, cur)
        else:
            cur = 0
        prev_time = t
    return float(best / len(ordered))


def longest_continuous_same_track_fraction(group: pd.DataFrame) -> float:
    group = group.sort_values("time")
    n = len(group)
    best = 0
    cur = 0
    prev_time = None
    prev_tid = None
    for row in group.itertuples(index=False):
        tid = row.target_tid
        ok = bool(row.matched) and pd.notna(tid)
        if not ok:
            cur = 0
            prev_time = None
            prev_tid = None
            continue
        tid = int(tid)
        t = int(row.time)
        if prev_tid == tid and prev_time is not None and t == prev_time + 1:
            cur += 1
        else:
            cur = 1
        best = max(best, cur)
        prev_tid = tid
        prev_time = t
    return float(best / n) if n else np.nan


def summarize_trajectories(match_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, gt_tid), group in match_df.groupby(["condition", "gt_tid"]):
        group = group.sort_values("time")
        matched = group[group["matched"] & group["target_tid"].notna()]
        target_ids = [int(x) for x in matched["target_tid"].tolist()]
        dominant_tid = np.nan
        dominant_fraction = 0.0
        if target_ids:
            counts = pd.Series(target_ids).value_counts()
            dominant_tid = int(counts.index[0])
            dominant_fraction = float(counts.iloc[0] / len(group))

        link_total = 0
        link_correct = 0
        prev = None
        switches = 0
        for row in group.itertuples(index=False):
            current = int(row.target_tid) if bool(row.matched) and pd.notna(row.target_tid) else None
            if prev is not None:
                prev_t, prev_tid = prev
                if int(row.time) == prev_t + 1:
                    link_total += 1
                    if current is not None and prev_tid is not None and current == prev_tid:
                        link_correct += 1
                    elif current is not None and prev_tid is not None and current != prev_tid:
                        switches += 1
            prev = (int(row.time), current)

        tf = longest_continuous_same_track_fraction(group)
        complete = bool(len(group) > 0 and group["matched"].all() and (link_total == 0 or link_correct == link_total))
        rows.append(
            {
                "condition": condition,
                "gt_tid": int(gt_tid),
                "n_gt_cells": int(len(group)),
                "detected_cells": int(group["matched"].sum()),
                "det_rate": float(group["matched"].mean()),
                "seg_mean_iou_all_gt": float(group["iou"].where(group["matched"], 0.0).mean()),
                "seg_mean_iou_matched": float(group.loc[group["matched"], "iou"].mean()) if group["matched"].any() else np.nan,
                "dominant_target_tid": dominant_tid,
                "dominant_fraction": dominant_fraction,
                "n_target_fragments": int(len(set(target_ids))),
                "identity_switches": int(switches),
                "link_total": int(link_total),
                "link_correct": int(link_correct),
                "lnk_rate": float(link_correct / link_total) if link_total else np.nan,
                "tf_fraction": tf,
                "complete_track": complete,
            }
        )
    return pd.DataFrame(rows)


def anchor_match_with_tolerance(
    match_df: pd.DataFrame,
    gt_tid: int,
    anchor_time: int,
    tolerance: int,
    direction: str,
) -> Tuple[Optional[int], float, Optional[int]]:
    if direction == "before":
        candidate_times = set(range(anchor_time - tolerance, anchor_time + 1))
    elif direction == "after":
        candidate_times = set(range(anchor_time, anchor_time + tolerance + 1))
    else:
        candidate_times = set(range(anchor_time - tolerance, anchor_time + tolerance + 1))
    sub = match_df[
        (match_df["gt_tid"].astype(int) == int(gt_tid))
        & (match_df["time"].astype(int).isin(candidate_times))
        & (match_df["matched"])
        & (match_df["target_tid"].notna())
    ]
    if sub.empty:
        return None, 0.0, None
    best = sub.sort_values("iou", ascending=False).iloc[0]
    return int(best["target_tid"]), float(best["iou"]), int(best["time"])


def summarize_branching(match_df: pd.DataFrame, edges_df: pd.DataFrame, tolerances: Sequence[int]) -> pd.DataFrame:
    rows = []
    condition = str(match_df["condition"].iloc[0]) if len(match_df) else ""
    for mother_id, group in edges_df.groupby("mother_gt_tid"):
        daughters = sorted(int(x) for x in group["daughter_gt_tid"].tolist())[:2]
        if len(daughters) < 2:
            continue
        mother_end = int(group["mother_end"].max())
        daughter_starts = {
            int(row["daughter_gt_tid"]): int(row["daughter_start"])
            for _, row in group.iterrows()
        }
        for tol in tolerances:
            mother_tid, mother_iou, mother_time = anchor_match_with_tolerance(
                match_df, int(mother_id), mother_end, tol, "before"
            )
            d1_tid, d1_iou, d1_time = anchor_match_with_tolerance(
                match_df, daughters[0], int(daughter_starts[daughters[0]]), tol, "after"
            )
            d2_tid, d2_iou, d2_time = anchor_match_with_tolerance(
                match_df, daughters[1], int(daughter_starts[daughters[1]]), tol, "after"
            )
            target_ids = [mother_tid, d1_tid, d2_tid]
            complete = all(x is not None for x in target_ids)
            distinct = complete and len(set(int(x) for x in target_ids if x is not None)) == 3
            rows.append(
                {
                    "condition": condition,
                    "mother_gt_tid": int(mother_id),
                    "daughter1_gt_tid": daughters[0],
                    "daughter2_gt_tid": daughters[1],
                    "tolerance": int(tol),
                    "mother_target_tid": mother_tid,
                    "daughter1_target_tid": d1_tid,
                    "daughter2_target_tid": d2_tid,
                    "mother_match_time": mother_time,
                    "daughter1_match_time": d1_time,
                    "daughter2_match_time": d2_time,
                    "mother_iou": mother_iou,
                    "daughter1_iou": d1_iou,
                    "daughter2_iou": d2_iou,
                    "complete_branch_anchors": bool(complete),
                    "distinct_branch_tracks": bool(distinct),
                    "bc_recovered": bool(distinct),
                }
            )
    return pd.DataFrame(rows)


def annotated_subgraph_tra_score(traj_df: pd.DataFrame, branch_df: pd.DataFrame) -> float:
    """AOGM-style TRA on the annotated GT lineage subgraph.

    This is not official full-movie CTC TRA because cells outside the annotated
    GT subgraph are ignored.  Within the annotated subgraph, it penalizes:
    missing GT nodes, missing consecutive temporal links, and missing/incorrect
    mother-daughter branch-anchor recovery errors.
    """
    if traj_df.empty:
        return np.nan
    node_total = float(traj_df["n_gt_cells"].sum())
    if "covered_cells_by_anchor_track" in traj_df.columns:
        node_correct = float(traj_df["covered_cells_by_anchor_track"].sum())
    elif "detected_cells" in traj_df.columns:
        node_correct = float(traj_df["detected_cells"].sum())
    else:
        node_correct = 0.0
    node_error = max(0.0, node_total - node_correct)

    link_total = float(traj_df["link_total"].sum()) if "link_total" in traj_df.columns else 0.0
    link_correct = float(traj_df["link_correct"].sum()) if "link_correct" in traj_df.columns else 0.0
    link_error = max(0.0, link_total - link_correct)

    exact_branch = branch_df[branch_df["tolerance"].astype(int) == 0].copy() if len(branch_df) else pd.DataFrame()
    branch_edge_total = 2.0 * float(len(exact_branch))
    branch_edge_correct = 0.0
    if len(exact_branch):
        mother_present = exact_branch["mother_target_tid"].notna()
        d1_present = exact_branch["daughter1_target_tid"].notna()
        d2_present = exact_branch["daughter2_target_tid"].notna()
        daughters_distinct = (
            (~d1_present)
            | (~d2_present)
            | (exact_branch["daughter1_target_tid"] != exact_branch["daughter2_target_tid"])
        )
        d1_edge = (
            mother_present
            & d1_present
            & daughters_distinct
            & (exact_branch["mother_target_tid"] != exact_branch["daughter1_target_tid"])
        )
        d2_edge = (
            mother_present
            & d2_present
            & daughters_distinct
            & (exact_branch["mother_target_tid"] != exact_branch["daughter2_target_tid"])
        )
        branch_edge_correct = float(d1_edge.sum() + d2_edge.sum())
    branch_edge_error = max(0.0, branch_edge_total - branch_edge_correct)

    empty_graph_cost = node_total + link_total + branch_edge_total
    if empty_graph_cost <= 0:
        return np.nan
    edit_cost = node_error + link_error + branch_edge_error
    return float(max(0.0, 1.0 - edit_cost / empty_graph_cost))


def summarize_condition(
    condition: str,
    match_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    branch_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    link_total = int(traj_df["link_total"].sum()) if len(traj_df) else 0
    link_correct = int(traj_df["link_correct"].sum()) if len(traj_df) else 0
    det = float(match_df["matched"].mean()) if len(match_df) else np.nan
    seg = float(match_df["iou"].where(match_df["matched"], 0.0).mean()) if len(match_df) else np.nan
    lnk = float(link_correct / link_total) if link_total else np.nan
    tf = float(traj_df["tf_fraction"].mean()) if len(traj_df) else np.nan
    ct = float(traj_df["complete_track"].mean()) if len(traj_df) else np.nan
    bc0 = float(branch_df.loc[branch_df["tolerance"] == 0, "bc_recovered"].mean()) if len(branch_df) else np.nan
    annotated_tra = annotated_subgraph_tra_score(traj_df, branch_df)
    base_metrics = [
        ("GT-region DET", det, "GT-region cell detection rate, IoU >= threshold"),
        ("GT-region SEG", seg, "Mean GT-region cell IoU; missed cells contribute zero"),
        ("Subgraph LNK", lnk, "Correct consecutive GT-subgraph links recovered by the same matched target track identity"),
        ("Subgraph TF", tf, "Mean longest continuous fraction covered in each GT-subgraph trajectory"),
        ("Subgraph CT", ct, "Fraction of GT-subgraph trajectories with all annotated cells and links recovered"),
        (
            "Subgraph TRA",
            annotated_tra,
            (
                "AOGM-style score on GT-subgraph nodes, temporal links, and branch-anchor recovery; "
                "not official full-movie CTC TRA"
            ),
        ),
    ]
    for metric, value, description in base_metrics:
        rows.append({"condition": condition, "metric": metric, "value": value, "description": description})
    for tol, group in branch_df.groupby("tolerance"):
        rows.append(
            {
                "condition": condition,
                "metric": f"Subgraph BC-anchor({int(tol)})",
                "value": float(group["bc_recovered"].mean()),
                "description": (
                    "Subgraph branching-anchor correctness: mother-last and daughter-start anchors "
                    "map to three distinct target tracks"
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_metric_bars(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    order = METHOD_ORDER
    colors = METHOD_COLORS
    metric_groups = [
        (
            "Tracking and segmentation quality",
            ["GT-region DET", "GT-region SEG", "Subgraph LNK", "Subgraph TF", "Subgraph CT", "Subgraph TRA"],
        ),
        ("Branching correctness", ["Subgraph BC-anchor(0)", "Subgraph BC-anchor(2)", "Subgraph BC-anchor(5)"]),
    ]
    metrics = [metric for _, group_metrics in metric_groups for metric in group_metrics]
    plot_df = summary_df[summary_df["metric"].isin(metrics)].copy()
    pivot = plot_df.pivot_table(index="metric", columns="condition", values="value", aggfunc="first").reindex(metrics)

    fig = plt.figure(figsize=(7.4, 5.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.66])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    width = 0.24
    legend_handles = None
    for ax, (group_title, group_metrics) in zip(axes, metric_groups):
        x = np.arange(len(group_metrics))
        all_vals = []
        for idx, condition in enumerate(order):
            vals = [
                pivot.loc[m, condition] * 100
                if condition in pivot.columns and pd.notna(pivot.loc[m, condition])
                else np.nan
                for m in group_metrics
            ]
            all_vals.extend([v for v in vals if pd.notna(v)])
            xpos_arr = x + (idx - 1.0) * width
            bars = ax.bar(
                xpos_arr,
                vals,
                width,
                label=condition,
                color=colors[condition],
                edgecolor="white",
                linewidth=0.5,
            )
            if legend_handles is None:
                legend_handles = bars
            for xpos, val in zip(xpos_arr, vals):
                if pd.notna(val):
                    label_y = val + 2.0 if val >= 8 else val + 3.2
                    ax.text(
                        xpos,
                        label_y,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        rotation=90,
                        color="#222222",
                    )

        ax.set_xticks(x)
        xlabels = [
            m.replace("GT-region ", "GT-region\n").replace("Subgraph ", "Subgraph\n")
            for m in group_metrics
        ]
        ax.set_xticklabels(xlabels)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, max(106, max(all_vals + [1]) * 1.20))
        ax.set_title(group_title, loc="left", fontweight="bold", pad=4)
        ax.grid(False)

    fig.suptitle(
        "Process2 CTC-style tracking comparison on hand-annotated lineage events",
        fontweight="bold",
        fontsize=11,
    )
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=colors[condition], edgecolor="white", label=METHOD_DISPLAY_LABELS.get(condition, condition))
        for condition in order
    ]
    axes[0].legend(
        handles=legend_handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        handlelength=1.5,
        columnspacing=1.8,
    )
    note = (
        "Partial-GT evaluation on annotated GT trajectories. DET/SEG use all evaluated GT cells; "
        "subgraph LNK/TF/CT/TRA use the annotated GT trajectory subgraph; BC-anchor uses division-anchor tolerance windows."
    )
    fig.text(0.01, -0.01, note, fontsize=7.3, color="#555555")
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(fig_dir / f"figSXX_process2_ctc_style_tracking_comparison.{ext}", dpi=450)
    plt.close(fig)


def plot_distributions(traj_df: pd.DataFrame, match_df: pd.DataFrame, branch_df: pd.DataFrame, fig_dir: Path) -> None:
    order = METHOD_ORDER
    colors = [METHOD_COLORS[c] for c in order]
    short_labels = METHOD_SHORT_LABELS

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.2), constrained_layout=True)

    tf_data = [traj_df.loc[traj_df["condition"] == c, "tf_fraction"].dropna().to_numpy() for c in order]
    bp = axes[0].boxplot(
        tf_data,
        tick_labels=short_labels,
        showfliers=False,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#222222", "linewidth": 1.2},
        whiskerprops={"color": "#333333", "linewidth": 0.9},
        capprops={"color": "#333333", "linewidth": 0.9},
    )
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.82)
        box.set_edgecolor("#333333")
        box.set_linewidth(0.8)
    axes[0].set_ylabel("Fraction")
    axes[0].set_ylim(-0.03, 1.04)
    axes[0].set_title("Continuous track recovery", fontweight="bold", loc="left")
    axes[0].grid(False)

    iou_source = match_df
    if "offset" in iou_source.columns:
        iou_source = iou_source[iou_source["offset"].astype(int) == 0]
    iou_data = [iou_source.loc[(iou_source["condition"] == c) & (iou_source["matched"]), "iou"].dropna().to_numpy() for c in order]
    bp = axes[1].boxplot(
        iou_data,
        tick_labels=short_labels,
        showfliers=False,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#222222", "linewidth": 1.2},
        whiskerprops={"color": "#333333", "linewidth": 0.9},
        capprops={"color": "#333333", "linewidth": 0.9},
    )
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.82)
        box.set_edgecolor("#333333")
        box.set_linewidth(0.8)
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0, 1.04)
    axes[1].set_title("Matched-mask quality", fontweight="bold", loc="left")
    axes[1].grid(False)

    bc = (
        branch_df.groupby(["condition", "tolerance"])["bc_recovered"]
        .mean()
        .reset_index()
    )
    tolerances = sorted(int(x) for x in bc["tolerance"].unique())
    x = np.arange(len(tolerances))
    width = 0.24
    for idx, condition in enumerate(order):
        vals = []
        for tol in tolerances:
            sub = bc[(bc["condition"] == condition) & (bc["tolerance"] == tol)]
            vals.append(float(sub["bc_recovered"].iloc[0]) * 100 if len(sub) else np.nan)
        axes[2].bar(
            x + (idx - 1.0) * width,
            vals,
            width,
            label=METHOD_DISPLAY_LABELS.get(condition, condition),
            color=colors[idx],
            edgecolor="white",
            linewidth=0.5,
        )
        for xpos, val in zip(x + (idx - 1.0) * width, vals):
            if pd.notna(val) and val > 0:
                axes[2].text(xpos, val + 2.2, f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"Subgraph\nBC-anchor({t})" for t in tolerances])
    axes[2].set_ylabel("Recovered divisions (%)")
    axes[2].set_ylim(0, 105)
    axes[2].set_title("Branching correctness", fontweight="bold", loc="left")
    axes[2].grid(False)
    axes[2].legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(0.0, 1.02))

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(True)
        ax.set_axisbelow(True)
    fig.suptitle("Per-event recovery distributions", fontweight="bold", fontsize=10.5)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(fig_dir / f"figSXX_process2_ctc_style_metric_distributions.{ext}", dpi=450)
    plt.close(fig)


def write_readme(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    lines = [
        "# Process2 CTC-style tracking comparison",
        "",
        "These figures compare SORT, LiveCellX, and Ultrack on the hand-annotated process2 GT lineage subset.",
        "",
        "Important limitation: the GT trajectory collection is partial. Because the ground truth contains only partial trajectories, we evaluated tracking continuity on the annotated GT trajectory subgraph rather than using the full official CTC graph-matching evaluation.",
        "",
        "For the TimeSformer correction, GT is only used to restrict inference to trajectories that overlap the annotated GT anchors, which saves runtime. The split frame is estimated from model predictions by scanning 8-frame trajectory windows and splitting at the frame immediately before the first non-mitosis window after a mitosis-positive run. Daughter A is the post-split part of the same trajectory, and daughter B is the nearest nearby post-division trajectory.",
        "",
        "## Criteria",
        "",
        "- GT-region DET: fraction of evaluated GT-region cells matched to a target object at the same frame with IoU >= 0.5.",
        "- GT-region SEG: mean IoU over evaluated GT-region cells; unmatched cells contribute 0.",
        "- Subgraph LNK: fraction of consecutive links inside the GT trajectory subgraph whose matched method cells remain connected by the same track identity.",
        "- Subgraph TF: longest continuous segment with recovered cells and temporal links, divided by the annotated GT-subgraph trajectory length.",
        "- Subgraph CT: fraction of GT-subgraph trajectories whose cells and consecutive temporal links are fully recovered inside the annotated span.",
        "- Subgraph BC-anchor(i): division/branching-anchor correctness with i-frame tolerance. Mother-last and two daughter-start anchors must map to three distinct target tracks. Because explicit parent-daughter graph edges are not checked here, this is an anchor-based BC proxy.",
        "- Subgraph TRA: AOGM-style score computed only on the annotated GT lineage subgraph. It penalizes missing annotated GT cells, broken temporal links, and missing/incorrect mother-daughter branch-anchor recovery. Cells outside the annotated subgraph are ignored.",
        "",
        "## Output files",
        "",
        "- `figSXX_process2_ctc_style_tracking_comparison.pdf/png/svg`: main bar-plot comparison for GT-region DET/SEG, subgraph LNK/TF/CT/TRA, and subgraph BC-anchor.",
        "- `figSXX_process2_ctc_style_metric_distributions.pdf/png/svg`: per-trajectory TF distribution, matched IoU distribution, and BC tolerance comparison.",
        "- `process2_ctc_style_cell_matches.csv`: one-to-one same-frame matches for all evaluated GT-region cells.",
        "- `process2_ctc_style_trajectory_metrics.csv`: per-GT-trajectory CTC-style metrics.",
        "- `process2_ctc_style_branching_metrics.csv`: per-division subgraph BC-anchor tolerance metrics.",
        "- `process2_ctc_style_summary_metrics.csv`: summary metric table used for the main figure.",
        "",
        "## Current summary",
        "",
    ]
    if not summary_df.empty:
        pivot = summary_df.pivot_table(index="metric", columns="condition", values="value", aggfunc="first")
        lines.append("```")
        lines.append(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
        lines.append("```")
        lines.append("")
    (fig_dir / "README_process2_ctc_style_tracking_comparison.md").write_text("\n".join(lines))


def plot_from_existing_outputs(args: argparse.Namespace) -> None:
    fig_dir = Path(args.output_dir)
    traj_df = pd.read_csv(fig_dir / "process2_ctc_style_trajectory_metrics.csv")
    match_df = pd.read_csv(fig_dir / "process2_ctc_style_cell_matches.csv")
    branch_df = pd.read_csv(fig_dir / "process2_ctc_style_branching_metrics.csv")
    summary_parts = []
    for condition in METHOD_ORDER:
        cond_match = match_df[match_df["condition"] == condition].copy()
        cond_traj = traj_df[traj_df["condition"] == condition].copy()
        cond_branch = branch_df[branch_df["condition"] == condition].copy()
        if len(cond_match) or len(cond_traj) or len(cond_branch):
            summary_parts.append(summarize_condition(condition, cond_match, cond_traj, cond_branch))
    summary_df = pd.concat(summary_parts, ignore_index=True)
    summary_df.to_csv(fig_dir / "process2_ctc_style_summary_metrics.csv", index=False)
    plot_metric_bars(summary_df, fig_dir)
    plot_distributions(traj_df, match_df, branch_df, fig_dir)
    write_readme(summary_df, fig_dir)
    print(f"[done] regenerated figures and README from existing CSVs in {fig_dir}")


def run_evaluation(args: argparse.Namespace) -> None:
    if args.plot_only_from_existing_csv:
        plot_from_existing_outputs(args)
        return

    fig_dir = Path(args.output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cfg = MatchConfig(iou_threshold=args.iou_threshold)

    gt_sctc = load_sctc(resolve_path(Path(args.gt_path)))
    gt_ids = relation_ids_from_gt(gt_sctc)
    edges_df = lineage_edges_from_gt(gt_sctc, gt_ids, min_frame=int(args.min_frame), max_frame=args.max_frame)
    full_gt_cell_df = build_gt_cell_table(gt_sctc, gt_ids, min_frame=int(args.min_frame), max_frame=args.max_frame)
    tolerance_max = max(args.branch_tolerances) if args.branch_tolerances else 0
    gt_cell_df = full_gt_cell_df
    branch_gt_cell_df = build_division_anchor_cell_table(
        gt_sctc, edges_df, tolerance_max=tolerance_max, min_frame=int(args.min_frame), max_frame=args.max_frame
    )
    scope = f"annotated GT trajectory-subgraph cells in frames {int(args.min_frame)}-{args.max_frame}"
    validate_gt_scope(gt_ids, edges_df, full_gt_cell_df, gt_cell_df)
    print(
        f"[gt] {len(gt_ids)} GT trajectories, {len(edges_df)} mother-daughter edges, "
        f"{len(full_gt_cell_df)} evaluation cells ({scope}), "
        f"{len(branch_gt_cell_df)} branch-anchor/tolerance cells"
    )
    full_gt_cell_df.to_csv(fig_dir / "process2_ctc_style_full_gt_cells.csv", index=False)
    gt_cell_df.to_csv(fig_dir / "process2_ctc_style_gt_cells.csv", index=False)
    branch_gt_cell_df.to_csv(fig_dir / "process2_ctc_style_branch_gt_cells.csv", index=False)
    edges_df.to_csv(fig_dir / "process2_ctc_style_gt_edges.csv", index=False)

    sort_after_eval_path = resolve_path(Path(args.sort_after_path))
    preloaded_sctcs: Dict[str, SingleCellTrajectoryCollection] = {}
    if args.run_timesformer_correction:
        sort_after_sctc = load_sctc(sort_after_eval_path)
        candidate_tids = None
        if not args.timesformer_all_trajectories:
            candidate_tids = select_timesformer_candidate_tids(sort_after_sctc, gt_sctc, edges_df, args)
        corrected_sctc, _, _ = apply_timesformer_lineage_correction(
            sort_after_sctc,
            candidate_tids,
            args,
            fig_dir,
        )
        sort_after_eval_path = Path(args.timesformer_output_sctc)
        preloaded_sctcs["LiveCellX"] = corrected_sctc
    elif args.replay_timesformer_predictions:
        sort_after_sctc = load_sctc(sort_after_eval_path)
        corrected_sctc, _, _ = apply_timesformer_lineage_correction_from_predictions(
            sort_after_sctc,
            Path(args.timesformer_prediction_csv),
            args,
            fig_dir,
        )
        sort_after_eval_path = Path(args.timesformer_output_sctc)
        preloaded_sctcs["LiveCellX"] = corrected_sctc
    elif args.use_existing_timesformer_correction:
        sort_after_eval_path = resolve_path(Path(args.timesformer_output_sctc))
        print(f"[timesformer] reusing corrected after-CSNet SCTC: {sort_after_eval_path}")
    else:
        print(
            f"[timesformer] no model inference requested; using --sort-after-path for the LiveCellX bar: "
            f"{sort_after_eval_path}"
        )

    write_benchmark_scope(fig_dir, args, gt_ids, edges_df, full_gt_cell_df, gt_cell_df, scope, branch_gt_cell_df)

    all_matches = []
    all_traj = []
    all_branch = []
    all_summary = []

    ultrack_eval_path = resolve_path(Path(args.ultrack_zarr_path))
    ultrack_kind = "sctc" if is_sctc_json_path(ultrack_eval_path) else "zarr"
    conditions = [
        ("SORT", "sctc", resolve_path(Path(args.sort_before_path), Path(args.sort_before_fallback_path))),
        ("LiveCellX", "sctc", sort_after_eval_path),
        ("Ultrack", ultrack_kind, ultrack_eval_path),
    ]

    for condition, kind, path in conditions:
        if kind == "sctc":
            target_sctc = preloaded_sctcs.get(condition)
            if target_sctc is None:
                target_sctc = load_sctc(path)
            match_df = match_sctc_condition(condition, target_sctc, gt_sctc, gt_cell_df, cfg)
            traj_df = summarize_trajectories(match_df)
            if condition not in preloaded_sctcs:
                del target_sctc
        else:
            match_df = match_ultrack_condition(condition, path, gt_sctc, gt_cell_df, cfg)
            traj_df = summarize_trajectories(match_df)

        branch_match_df = attach_branch_anchor_matches(condition, match_df, branch_gt_cell_df)
        branch_df = summarize_branching(branch_match_df, edges_df, args.branch_tolerances)
        summary_df = summarize_condition(condition, match_df, traj_df, branch_df)

        all_matches.append(match_df)
        all_traj.append(traj_df)
        all_branch.append(branch_df)
        all_summary.append(summary_df)

    match_df = pd.concat(all_matches, ignore_index=True)
    traj_df = pd.concat(all_traj, ignore_index=True)
    branch_df = pd.concat(all_branch, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)

    match_df.to_csv(fig_dir / "process2_ctc_style_cell_matches.csv", index=False)
    traj_df.to_csv(fig_dir / "process2_ctc_style_trajectory_metrics.csv", index=False)
    branch_df.to_csv(fig_dir / "process2_ctc_style_branching_metrics.csv", index=False)
    summary_df.to_csv(fig_dir / "process2_ctc_style_summary_metrics.csv", index=False)

    plot_metric_bars(summary_df, fig_dir)
    plot_distributions(traj_df, match_df, branch_df, fig_dir)
    write_readme(summary_df, fig_dir)
    print(f"[done] saved figures and tables to {fig_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-path", default=str(GT_PATH))
    parser.add_argument("--sort-before-path", default=str(SORT_BEFORE_PATH))
    parser.add_argument("--sort-before-fallback-path", default=str(SORT_BEFORE_FALLBACK_PATH))
    parser.add_argument("--sort-after-path", default=str(SORT_AFTER_PATH))
    parser.add_argument(
        "--ultrack-path",
        "--ultrack-zarr-path",
        dest="ultrack_zarr_path",
        default=str(ULTRACK_ZARR_PATH),
        help=(
            "Ultrack input for evaluation. The default is the requested exp4 GT-region-only SCTC JSON. "
            "A raw zarr label folder is also accepted."
        ),
    )
    parser.add_argument("--output-dir", default=str(FIG_DIR))
    parser.add_argument("--min-frame", type=int, default=0, help="First frame included in GT evaluation.")
    parser.add_argument("--max-frame", type=int, default=99, help="Last frame included in GT evaluation; default keeps frames 0-99.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--branch-tolerances", type=int, nargs="+", default=[0, 2, 5])
    parser.add_argument(
        "--plot-only-from-existing-csv",
        action="store_true",
        help="Regenerate figures/README from existing process2_ctc_style_*.csv files without rerunning matching.",
    )
    parser.add_argument(
        "--full-cell-matching",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Full annotated GT-cell matching is now always used for "
            "GT-region DET/SEG and subgraph LNK/TF/CT/TRA."
        ),
    )
    parser.add_argument(
        "--run-timesformer-correction",
        action="store_true",
        help=(
            "Run TimeSformer on after-CSNet trajectories selected by GT-anchor overlap, split predicted "
            "mitotic trajectories, save the corrected SCTC, and use it for the after-CSNet bar."
        ),
    )
    parser.add_argument(
        "--replay-timesformer-predictions",
        action="store_true",
        help=(
            "Skip model inference and rebuild the corrected after-CSNet SCTC from the saved "
            "process2_timesformer_window_predictions.csv using the current split/daughter logic."
        ),
    )
    parser.add_argument(
        "--use-existing-timesformer-correction",
        action="store_true",
        help="Skip model inference and use --timesformer-output-sctc for the after-CSNet + TimeSformer bar.",
    )
    parser.add_argument("--timesformer-config-path", default=str(TIMESFORMER_CONFIG_PATH))
    parser.add_argument("--timesformer-model-path", default=str(TIMESFORMER_MODEL_PATH))
    parser.add_argument("--timesformer-output-sctc", default=str(TIMESFORMER_CORRECTED_AFTER_PATH))
    parser.add_argument("--timesformer-prediction-csv", default=str(TIMESFORMER_WINDOW_PREDICTIONS_PATH))
    parser.add_argument(
        "--save-timesformer-corrected-sctc",
        action="store_true",
        help=(
            "Also write the full corrected after-CSNet trajectory collection JSON. "
            "This can be very memory- and disk-heavy, so it is off by default."
        ),
    )
    parser.add_argument(
        "--timesformer-video-dir",
        default="/tmp/livecellx_process2_timesformer_window_videos",
        help="Temporary directory used for generated trajectory-window MP4 clips.",
    )
    parser.add_argument("--timesformer-device", default="cuda:0")
    parser.add_argument("--timesformer-window-size", type=int, default=8)
    parser.add_argument("--timesformer-step-size", type=int, default=1)
    parser.add_argument("--timesformer-padding", type=int, default=200)
    parser.add_argument("--timesformer-frame-type", default="combined", choices=["video", "mask", "combined"])
    parser.add_argument(
        "--timesformer-division-label",
        type=int,
        default=0,
        help="Classifier label index treated as mitosis/division.",
    )
    parser.add_argument("--timesformer-max-gap", type=int, default=1)
    parser.add_argument("--timesformer-min-positive-windows", type=int, default=1)
    parser.add_argument(
        "--timesformer-all-trajectories",
        action="store_true",
        help="Run TimeSformer on every after-CSNet trajectory instead of only GT-anchor-overlapping tracks.",
    )
    parser.add_argument(
        "--timesformer-max-trajectories",
        type=int,
        default=0,
        help="Debug limit for TimeSformer correction. 0 means no limit.",
    )
    parser.add_argument("--daughter-search-window", type=int, default=8)
    parser.add_argument("--daughter-max-distance", type=float, default=180.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
