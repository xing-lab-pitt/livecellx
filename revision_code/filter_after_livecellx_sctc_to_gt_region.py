#!/usr/bin/env python
"""Create a method SCTC restricted to the annotated GT mask region.

The output is a normal SingleCellTrajectoryCollection JSON: it can be read with
SingleCellTrajectoryCollection.load_from_json_file and opened by
create_sctc_edit_viewer_by_interval.

Important: this is a spatial GT-region filter, not a one-best-cell matching
evaluation.  For each GT cell mask at each frame, every same-frame target cell
whose mask has pixels inside that GT cell region is kept.  The output trajectory
collection contains only those selected target cell timepoints, grouped by their
original target trajectory IDs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from livecellx.core.single_cell import SingleCellTrajectory, SingleCellTrajectoryCollection

from process2_ctc_style_tracking_evaluation import (
    GT_PATH,
    MatchConfig,
    RESULTS_DIR,
    TIMESFORMER_CORRECTED_AFTER_PATH,
    bbox_area,
    bbox_intersects,
    build_gt_cell_table,
    build_time_index,
    load_sctc,
    progress_iter,
    relation_ids_from_gt,
    resolve_path,
)


DEFAULT_OUTPUT = (
    RESULTS_DIR
    / "ground_truth_trajs"
    / "sctc-final-2026-06-02_before_correction_timesformer_lineage_corrected_gt_region_only.json"
)


def bbox_iou(a, b) -> float:
    y0 = max(float(a[0]), float(b[0]))
    x0 = max(float(a[1]), float(b[1]))
    y1 = min(float(a[2]), float(b[2]))
    x1 = min(float(a[3]), float(b[3]))
    if y1 <= y0 or x1 <= x0:
        return 0.0
    inter = (y1 - y0) * (x1 - x0)
    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def _intersection_bbox(a, b) -> Tuple[np.ndarray | None, float]:
    y0 = max(float(a[0]), float(b[0]))
    x0 = max(float(a[1]), float(b[1]))
    y1 = min(float(a[2]), float(b[2]))
    x1 = min(float(a[3]), float(b[3]))
    if y1 <= y0 or x1 <= x0:
        return None, 0.0
    bbox = np.asarray([np.floor(y0), np.floor(x0), np.ceil(y1), np.ceil(x1)], dtype=int)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None, 0.0
    return bbox, float((y1 - y0) * (x1 - x0))


def mask_overlap_pixels(gt_sc, target_sc) -> int:
    """Return true mask-pixel overlap in the shared bbox."""

    if gt_sc.bbox is None or target_sc.bbox is None:
        return 0
    bbox, _ = _intersection_bbox(gt_sc.bbox, target_sc.bbox)
    if bbox is None:
        return 0
    try:
        gt_mask = gt_sc.get_contour_mask(bbox=bbox, crop=True, dtype=bool)
        target_mask = target_sc.get_contour_mask(bbox=bbox, crop=True, dtype=bool)
    except Exception:
        return 0
    if gt_mask.shape != target_mask.shape:
        h = min(gt_mask.shape[0], target_mask.shape[0])
        w = min(gt_mask.shape[1], target_mask.shape[1])
        if h <= 0 or w <= 0:
            return 0
        gt_mask = gt_mask[:h, :w]
        target_mask = target_mask[:h, :w]
    return int(np.logical_and(gt_mask, target_mask).sum())


def _clip_bbox(bbox, shape) -> np.ndarray | None:
    clipped = np.asarray(
        [
            max(0, int(np.floor(bbox[0]))),
            max(0, int(np.floor(bbox[1]))),
            min(int(shape[0]), int(np.ceil(bbox[2]))),
            min(int(shape[1]), int(np.ceil(bbox[3]))),
        ],
        dtype=int,
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def build_gt_union_mask_for_time(gt_sctc: SingleCellTrajectoryCollection, rows: pd.DataFrame, shape) -> np.ndarray:
    union = np.zeros(tuple(shape[:2]), dtype=bool)
    for row in rows.to_dict("records"):
        gt_tid = int(row["gt_tid"])
        t = int(row["time"])
        gt_sc = gt_sctc.get_trajectory(gt_tid).get_sc(t)
        if gt_sc.bbox is None:
            continue
        bbox = _clip_bbox(gt_sc.bbox, union.shape)
        if bbox is None:
            continue
        try:
            mask = gt_sc.get_contour_mask(bbox=bbox, crop=True, dtype=bool)
        except Exception:
            continue
        h = min(mask.shape[0], bbox[2] - bbox[0])
        w = min(mask.shape[1], bbox[3] - bbox[1])
        if h <= 0 or w <= 0:
            continue
        union[bbox[0] : bbox[0] + h, bbox[1] : bbox[1] + w] |= mask[:h, :w]
    return union


def target_cells_inside_gt_region(
    gt_sctc: SingleCellTrajectoryCollection,
    target_sctc: SingleCellTrajectoryCollection,
    gt_cell_df: pd.DataFrame,
) -> pd.DataFrame:
    """Find every target cell/timepoint with actual mask pixels in GT mask regions."""

    allowed_times = set(int(x) for x in gt_cell_df["time"].unique())
    target_by_time = build_time_index(target_sctc, allowed_times)
    rows = []
    grouped = gt_cell_df.groupby("time", sort=True)
    for t, group in progress_iter(list(grouped), total=len(grouped), desc="filter target cells by GT masks"):
        t = int(t)
        shape = None
        if target_by_time.get(t):
            try:
                shape = target_by_time[t][0].sc.get_img_shape()
            except Exception:
                shape = None
        if shape is None:
            try:
                first_gt = gt_sctc.get_trajectory(int(group.iloc[0]["gt_tid"])).get_sc(t)
                shape = first_gt.get_img_shape()
            except Exception:
                shape = None
        if shape is None:
            rows.append(
                {
                    "time": int(t),
                    "target_tid": None,
                    "target_time": int(t),
                    "target_sc_id": "",
                    "overlap_pixels": 0,
                    "matched": False,
                    "reason": "no_image_shape",
                }
            )
            continue

        gt_union = build_gt_union_mask_for_time(gt_sctc, group, shape)
        matched_in_frame = False
        for cand in target_by_time.get(t, []):
            bbox = _clip_bbox(cand.bbox, gt_union.shape)
            if bbox is None:
                continue
            gt_crop = gt_union[bbox[0] : bbox[2], bbox[1] : bbox[3]]
            if not np.any(gt_crop):
                continue
            try:
                target_mask = cand.sc.get_contour_mask(bbox=bbox, crop=True, dtype=bool)
            except Exception:
                continue
            h = min(gt_crop.shape[0], target_mask.shape[0])
            w = min(gt_crop.shape[1], target_mask.shape[1])
            if h <= 0 or w <= 0:
                continue
            overlap_px = int(np.logical_and(gt_crop[:h, :w], target_mask[:h, :w]).sum())
            if overlap_px <= 0:
                continue
            matched_in_frame = True
            rows.append(
                {
                    "time": int(t),
                    "gt_cells_in_frame": int(len(group)),
                    "target_tid": int(cand.tid),
                    "target_time": int(t),
                    "target_sc_id": str(getattr(cand.sc, "id", "")),
                    "overlap_pixels": int(overlap_px),
                    "matched": True,
                    "reason": "",
                }
            )
        if not matched_in_frame:
            rows.append(
                {
                    "time": int(t),
                    "gt_cells_in_frame": int(len(group)),
                    "target_tid": None,
                    "target_time": int(t),
                    "target_sc_id": "",
                    "overlap_pixels": 0,
                    "matched": False,
                    "reason": "no_target_cell_inside_gt_region",
                }
            )
    return pd.DataFrame(rows)


def prune_lineage_links(sctc: SingleCellTrajectoryCollection, kept_ids: Set[int]) -> None:
    """Remove mother/daughter references to trajectories outside the filtered SCTC."""

    for tid, traj in sctc:
        mothers = {
            int(t.track_id): t
            for t in (getattr(traj, "mother_trajectories", set()) or set())
            if int(t.track_id) in kept_ids
        }
        daughters = {
            int(t.track_id): t
            for t in (getattr(traj, "daughter_trajectories", set()) or set())
            if int(t.track_id) in kept_ids
        }
        meta = traj.meta or {}
        for mother_id in meta.get(SingleCellTrajectory.META_MOTHER_IDS, []) or []:
            mother_id = int(mother_id)
            if mother_id in kept_ids:
                mothers[mother_id] = sctc.get_trajectory(mother_id)
        for daughter_id in meta.get(SingleCellTrajectory.META_DAUGHTER_IDS, []) or []:
            daughter_id = int(daughter_id)
            if daughter_id in kept_ids:
                daughters[daughter_id] = sctc.get_trajectory(daughter_id)

        traj.mother_trajectories = set(mothers.values())
        traj.daughter_trajectories = set(daughters.values())
        traj.meta[SingleCellTrajectory.META_MOTHER_IDS] = sorted(mothers)
        traj.meta[SingleCellTrajectory.META_DAUGHTER_IDS] = sorted(daughters)


def build_filtered_sctc(target_sctc: SingleCellTrajectoryCollection, match_df: pd.DataFrame) -> SingleCellTrajectoryCollection:
    kept_times: Dict[int, Set[int]] = defaultdict(set)
    matched = match_df[match_df["matched"] & match_df["target_tid"].notna()].copy()
    for _, row in matched.iterrows():
        kept_times[int(row["target_tid"])].add(int(row["target_time"]))

    filtered = SingleCellTrajectoryCollection()
    for tid in sorted(kept_times):
        if tid in target_sctc.track_id_to_trajectory:
            src = target_sctc.get_trajectory(tid)
            timeframe_to_single_cell = {
                int(t): sc
                for t, sc in sorted(src.timeframe_to_single_cell.items())
                if int(t) in kept_times[tid]
            }
            if timeframe_to_single_cell:
                traj = SingleCellTrajectory(
                    track_id=int(tid),
                    timeframe_to_single_cell=timeframe_to_single_cell,
                    img_dataset=src.img_dataset,
                    meta=dict(src.meta or {}),
                )
                filtered.add_trajectory(traj)
    prune_lineage_links(filtered, set(int(x) for x in filtered.track_id_to_trajectory.keys()))
    return filtered


def write_readme(
    path: Path,
    gt_path: Path,
    target_path: Path,
    output_path: Path,
    match_df: pd.DataFrame,
    kept_ids: Set[int],
    gt_cell_count: int,
) -> None:
    matched = match_df[match_df["matched"] & match_df["target_tid"].notna()]
    lines = [
        "# GT-mask-region filtered SCTC",
        "",
        f"GT SCTC: `{gt_path}`",
        f"Source after-LiveCellX SCTC: `{target_path}`",
        f"Filtered SCTC: `{output_path}`",
        "",
        "Definition: for each GT cell mask at each frame, keep every same-frame target cell whose mask has pixels inside that GT mask region.",
        "The filtered SCTC contains only those selected target cell timepoints, grouped by original target trajectory ID; it does not keep full outside-region trajectories.",
        "Lineage links are preserved only when both linked trajectories are kept, so loading the filtered JSON does not contain broken mother/daughter references.",
        "",
        f"GT-region cells defining the spatial region: {gt_cell_count}",
        f"Kept target cell rows: {int(match_df['matched'].sum())}",
        f"Unique kept target cell timepoints: {len(matched[['target_tid', 'target_time']].drop_duplicates()) if len(matched) else 0}",
        f"Kept target trajectories: {len(kept_ids)}",
        "Criterion: actual mask-pixel overlap > 0 inside the same-frame GT mask region.",
        "",
        "Use:",
        "```python",
        "from livecellx.core.single_cell import SingleCellTrajectoryCollection",
        f"traj_collection = SingleCellTrajectoryCollection.load_from_json_file('{output_path}')",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-path", default=str(GT_PATH))
    parser.add_argument("--target-path", default=str(TIMESFORMER_CORRECTED_AFTER_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Deprecated; ignored for GT-mask-region filtering.")
    parser.add_argument(
        "--keep-any-overlap",
        action="store_true",
        help="Keep a target trajectory if it has any same-frame bbox overlap with a GT-region cell, not only IoU >= threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = resolve_path(Path(args.gt_path))
    target_path = resolve_path(Path(args.target_path))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gt_sctc = load_sctc(gt_path)
    gt_ids = relation_ids_from_gt(gt_sctc)
    gt_cell_df = build_gt_cell_table(gt_sctc, gt_ids)
    if gt_cell_df.empty:
        raise ValueError("No GT-region cells found from mother/daughter GT trajectories.")

    print(f"[gt] {len(gt_ids)} annotated lineage trajectories, {len(gt_cell_df)} GT-region cells")
    target_sctc = load_sctc(target_path)
    match_df = target_cells_inside_gt_region(
        gt_sctc=gt_sctc,
        target_sctc=target_sctc,
        gt_cell_df=gt_cell_df,
    )
    kept_ids = {
        int(x)
        for x in match_df.loc[match_df["matched"] & match_df["target_tid"].notna(), "target_tid"].tolist()
    }
    if not kept_ids:
        raise ValueError("No target trajectories overlapped the GT region. Check input files and IoU threshold.")

    filtered = build_filtered_sctc(target_sctc, match_df)
    dataset_dir = output_path.parent / "datasets_gt_region_only"
    filtered.write_json(str(output_path), dataset_json_dir=dataset_dir)

    match_csv = output_path.with_suffix(".matched_gt_region_cells.csv")
    summary_txt = output_path.with_suffix(".README.md")
    match_df.to_csv(match_csv, index=False)
    write_readme(
        path=summary_txt,
        gt_path=gt_path,
        target_path=target_path,
        output_path=output_path,
        match_df=match_df,
        kept_ids=kept_ids,
        gt_cell_count=len(gt_cell_df),
    )
    print(
        f"[done] wrote {len(filtered)} trajectories and {len(filtered.get_all_scs())} cells to {output_path}"
    )
    print(f"[done] match table: {match_csv}")
    print(f"[done] readme: {summary_txt}")


if __name__ == "__main__":
    main()
