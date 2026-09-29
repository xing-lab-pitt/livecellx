#!/usr/bin/env python
"""Rebuild Ultrack SCTC JSON files with correct lineage IDs.

Problem this fixes
------------------
The Ultrack zarr label image and Ultrack `to_tracks_layer` table can use
non-identical numeric IDs.  The previous patch inserted the Ultrack lineage graph
(child track ID -> parent track ID) directly into SCTC trajectories that were
built by zarr label value.  That assigned mother/daughter relationships to the
wrong SCTC trajectories.

What this script does
---------------------
1. Read the Ultrack `tracks_df` exported from `to_tracks_layer`, where
   `track_id` and `parent_track_id` define the correct Ultrack trajectory graph.
2. For each `(track_id, frame, y, x)` row, read the Ultrack zarr frame and find
   the label mask at/near the Ultrack centroid.
3. Convert that label mask into a SingleCellStatic and group cells by the
   Ultrack `track_id` from `tracks_df`, not by the zarr label value.
4. Add mother-daughter links using the Ultrack graph IDs, now matching the SCTC
   trajectory IDs.
5. Write two loadable SCTC files:
   - full Ultrack SCTC
   - GT-region-only Ultrack SCTC, retaining only track/time masks that overlap
     the annotated GT mask regions.

This script changes only the Ultrack trajectory collections. It does not change
metric definitions or plotting logic. If the zarr and tracks_df are mismatched,
the script stops rather than creating a fake lineage collection.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from livecellx.core.single_cell import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection
from livecellx.core.sc_key_manager import SingleCellMetaKeyManager
from livecellx.segment.ou_simulator import find_contours_opencv

from process2_ctc_style_tracking_evaluation import (
    FINAL_TRAJ_COLLECTION_DIR,
    GT_PATH,
    ULTRACK_EXP4_ZARR_PATH,
    ZarrV3Labels,
    build_gt_cell_table,
    load_sctc,
    progress_iter,
    relation_ids_from_gt,
    resolve_path,
)

DEFAULT_TRACKS_DF = ULTRACK_EXP4_ZARR_PATH / "tracks_df.csv"
DEFAULT_GRAPH_JSON = ULTRACK_EXP4_ZARR_PATH / "graph.json"
DEFAULT_FULL_OUTPUT = FINAL_TRAJ_COLLECTION_DIR / "ultrack_from_cellpose_process2_beforecsnet_exp4_full.json"
DEFAULT_GT_REGION_OUTPUT = FINAL_TRAJ_COLLECTION_DIR / "ultrack_from_cellpose_process2_beforecsnet_exp4_gt_region_only.json"


def audit_tracks_df_zarr_pair(
    tracks_df: pd.DataFrame,
    zarr_path: Path,
    sample_n: int = 5000,
    min_centroid_track_id_match_rate: float = 0.95,
) -> dict:
    """Fail fast unless the zarr labels look like they came from `tracks_df`."""
    labels = ZarrV3Labels(resolve_path(zarr_path))
    frame_counts = []
    for t in progress_iter(range(labels.shape[0]), total=labels.shape[0], desc="audit zarr/tracks_df"):
        frame = labels.read_frame(int(t))
        zarr_count = int((np.unique(frame) != 0).sum())
        tracks_count = int((tracks_df["t"] == int(t)).sum())
        frame_counts.append((int(t), zarr_count, tracks_count))
    frame_counts_df = pd.DataFrame(frame_counts, columns=["frame", "zarr_count", "tracks_df_count"])
    count_diffs = frame_counts_df["zarr_count"] - frame_counts_df["tracks_df_count"]

    sample = tracks_df.sample(min(sample_n, len(tracks_df)), random_state=0) if len(tracks_df) else tracks_df
    n_valid = 0
    n_equal = 0
    n_nonzero = 0
    for row in sample.to_dict("records"):
        t = int(row["t"])
        y = int(round(float(row["y"])))
        x = int(round(float(row["x"])))
        track_id = int(row["track_id"])
        if t < 0 or t >= labels.shape[0] or y < 0 or y >= labels.shape[1] or x < 0 or x >= labels.shape[2]:
            continue
        frame = labels.read_frame(t)
        value = int(frame[y, x])
        n_valid += 1
        n_nonzero += int(value > 0)
        n_equal += int(value == track_id)

    match_rate = float(n_equal / n_valid) if n_valid else 0.0
    nonzero_rate = float(n_nonzero / n_valid) if n_valid else 0.0
    stats = {
        "zarr_path": str(resolve_path(zarr_path)),
        "tracks_df_rows": int(len(tracks_df)),
        "tracks_df_unique_track_ids": int(tracks_df["track_id"].nunique()),
        "zarr_total_label_timepoints": int(frame_counts_df["zarr_count"].sum()),
        "mean_frame_count_difference_zarr_minus_tracks_df": float(count_diffs.mean()),
        "min_frame_count_difference_zarr_minus_tracks_df": int(count_diffs.min()),
        "max_frame_count_difference_zarr_minus_tracks_df": int(count_diffs.max()),
        "centroid_sample_rows": int(n_valid),
        "centroid_nonzero_rate": nonzero_rate,
        "centroid_label_equals_track_id_rate": match_rate,
    }
    if not (count_diffs == 0).all() or match_rate < min_centroid_track_id_match_rate:
        raise RuntimeError(
            "Ultrack zarr and tracks_df are not the same run; refusing to rebuild lineage SCTC. "
            f"Audit stats: {json.dumps(stats, sort_keys=True)}"
        )
    print(f"[audit-pass] {json.dumps(stats, sort_keys=True)}")
    return stats


def largest_contour(contours: List[np.ndarray]) -> Optional[np.ndarray]:
    if not contours:
        return None
    try:
        import cv2

        return max(contours, key=lambda c: float(cv2.contourArea(c.astype(np.float32))))
    except Exception:
        return max(contours, key=len)


def frame_label_slices(frame: np.ndarray) -> Dict[int, Tuple[slice, slice]]:
    try:
        from scipy import ndimage as ndi

        objs = ndi.find_objects(frame)
        return {int(idx): obj for idx, obj in enumerate(objs, start=1) if obj is not None}
    except Exception:
        out = {}
        for label_value in np.unique(frame):
            label_value = int(label_value)
            if label_value == 0:
                continue
            ys, xs = np.where(frame == label_value)
            if len(ys) == 0:
                continue
            out[label_value] = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
        return out


def contour_for_label_slice(frame: np.ndarray, label_value: int, label_slice: Tuple[slice, slice]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    ys, xs = label_slice
    bbox = np.asarray([ys.start, xs.start, ys.stop, xs.stop], dtype=int)
    crop = (frame[label_slice] == int(label_value)).astype(np.uint8)
    area = int(crop.sum())
    if area <= 0:
        return None, None, 0
    contours = find_contours_opencv(crop)
    contour = largest_contour(contours)
    if contour is None or len(contour) == 0:
        return None, None, area
    contour = np.asarray(contour, dtype=float)
    contour[:, 0] += bbox[0]
    contour[:, 1] += bbox[1]
    return contour, bbox, area


def nearest_nonzero_label(frame: np.ndarray, y: float, x: float, radius: int = 12) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    cy = int(round(float(y)))
    cx = int(round(float(x)))
    if 0 <= cy < h and 0 <= cx < w:
        label = int(frame[cy, cx])
        if label > 0:
            return label, 0
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    crop = frame[y0:y1, x0:x1]
    coords = np.argwhere(crop > 0)
    if len(coords) == 0:
        return 0, -1
    yy = coords[:, 0] + y0
    xx = coords[:, 1] + x0
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    idx = int(np.argmin(dist2))
    return int(frame[int(yy[idx]), int(xx[idx])]), int(round(float(np.sqrt(dist2[idx]))))


def gt_mask_bbox(gt_sc, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if gt_sc.bbox is None:
        return None
    bbox = np.asarray(
        [
            max(0, int(np.floor(gt_sc.bbox[0]))),
            max(0, int(np.floor(gt_sc.bbox[1]))),
            min(int(frame_shape[0]), int(np.ceil(gt_sc.bbox[2]))),
            min(int(frame_shape[1]), int(np.ceil(gt_sc.bbox[3]))),
        ],
        dtype=int,
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def build_gt_union_masks(gt_sctc: SingleCellTrajectoryCollection, gt_cell_df: pd.DataFrame, frame_shape: Tuple[int, int]) -> Dict[int, np.ndarray]:
    masks: Dict[int, np.ndarray] = {}
    grouped = list(gt_cell_df.groupby("time", sort=True))
    for t, group in progress_iter(grouped, total=len(grouped), desc="build GT-region masks"):
        t = int(t)
        union = np.zeros(frame_shape, dtype=bool)
        for row in group.to_dict("records"):
            gt_tid = int(row["gt_tid"])
            try:
                gt_sc = gt_sctc.get_trajectory(gt_tid).get_sc(t)
            except Exception:
                continue
            bbox = gt_mask_bbox(gt_sc, frame_shape)
            if bbox is None:
                continue
            try:
                gt_mask = gt_sc.get_contour_mask(bbox=bbox, crop=True, dtype=bool)
            except Exception:
                continue
            h = min(bbox[2] - bbox[0], gt_mask.shape[0])
            w = min(bbox[3] - bbox[1], gt_mask.shape[1])
            if h > 0 and w > 0:
                union[bbox[0] : bbox[0] + h, bbox[1] : bbox[1] + w] |= gt_mask[:h, :w]
        masks[t] = union
    return masks


def sync_relation_meta(traj: SingleCellTrajectory) -> None:
    mothers = sorted(int(t.track_id) for t in (traj.mother_trajectories or set()))
    daughters = sorted(int(t.track_id) for t in (traj.daughter_trajectories or set()))
    traj.meta[SingleCellTrajectory.META_MOTHER_IDS] = mothers
    traj.meta[SingleCellTrajectory.META_DAUGHTER_IDS] = daughters


def add_lineage_links(sctc: SingleCellTrajectoryCollection, graph: Dict[int, int]) -> dict:
    for _, traj in sctc:
        traj.mother_trajectories = set()
        traj.daughter_trajectories = set()
        traj.meta[SingleCellTrajectory.META_MOTHER_IDS] = []
        traj.meta[SingleCellTrajectory.META_DAUGHTER_IDS] = []

    kept = 0
    skipped = 0
    for child_id, parent_id in sorted(graph.items()):
        child_id = int(child_id)
        parent_id = int(parent_id)
        if child_id == parent_id:
            skipped += 1
            continue
        if child_id not in sctc.track_id_to_trajectory or parent_id not in sctc.track_id_to_trajectory:
            skipped += 1
            continue
        child = sctc.get_trajectory(child_id)
        parent = sctc.get_trajectory(parent_id)
        child.add_mother(parent)
        parent.add_daughter(child)
        kept += 1

    mother_count = 0
    daughter_count = 0
    mothers_ge2 = 0
    for _, traj in sctc:
        sync_relation_meta(traj)
        if traj.meta[SingleCellTrajectory.META_DAUGHTER_IDS]:
            mother_count += 1
            if len(traj.meta[SingleCellTrajectory.META_DAUGHTER_IDS]) >= 2:
                mothers_ge2 += 1
        if traj.meta[SingleCellTrajectory.META_MOTHER_IDS]:
            daughter_count += 1
    return {
        "kept_edges": int(kept),
        "skipped_edges": int(skipped),
        "n_mothers_with_daughters": int(mother_count),
        "n_daughters_with_mother": int(daughter_count),
        "n_mothers_with_two_or_more_daughters": int(mothers_ge2),
    }


def build_sctc_from_tracks_df(
    tracks_df: pd.DataFrame,
    graph: Dict[int, int],
    zarr_path: Path,
    img_dataset,
    gt_union_masks: Optional[Dict[int, np.ndarray]] = None,
    source_name: str = "ultrack_tracks_df",
) -> Tuple[SingleCellTrajectoryCollection, pd.DataFrame, dict]:
    labels = ZarrV3Labels(resolve_path(zarr_path))
    track_to_cells: Dict[int, Dict[int, SingleCellStatic]] = defaultdict(dict)
    rows = []

    tracks_df = tracks_df.copy()
    tracks_df["track_id"] = tracks_df["track_id"].astype(int)
    tracks_df["t"] = tracks_df["t"].astype(int)
    grouped = list(tracks_df.groupby("t", sort=True))

    for t, group in progress_iter(grouped, total=len(grouped), desc=f"build {source_name}"):
        t = int(t)
        frame = labels.read_frame(t)
        label_slices = frame_label_slices(frame)
        gt_union = None if gt_union_masks is None else gt_union_masks.get(t)
        contour_cache: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray], int]] = {}
        for row in group.to_dict("records"):
            graph_tid = int(row["track_id"])
            y = float(row["y"])
            x = float(row["x"])
            centroid_label, centroid_dist = nearest_nonzero_label(frame, y, x)
            label_value = graph_tid
            if label_value not in label_slices:
                rows.append({
                    "track_id": graph_tid,
                    "time": t,
                    "converted": False,
                    "kept": False,
                    "zarr_label": int(centroid_label),
                    "expected_zarr_label": int(graph_tid),
                    "reason": "track_id_label_absent_in_frame",
                    "centroid_label_distance": centroid_dist,
                })
                continue
            if label_value not in contour_cache:
                label_slice = label_slices.get(int(label_value))
                if label_slice is None:
                    contour_cache[label_value] = (None, None, 0)
                else:
                    contour_cache[label_value] = contour_for_label_slice(frame, int(label_value), label_slice)
            contour, bbox, area = contour_cache[label_value]
            if contour is None or bbox is None:
                rows.append({"track_id": graph_tid, "time": t, "converted": False, "kept": False, "zarr_label": int(label_value), "expected_zarr_label": int(graph_tid), "reason": "empty_contour", "centroid_label_distance": centroid_dist})
                continue
            overlap_pixels = np.nan
            if gt_union is not None:
                overlap_pixels = int(np.logical_and(frame == int(label_value), gt_union).sum())
                if overlap_pixels <= 0:
                    rows.append({"track_id": graph_tid, "time": t, "converted": True, "kept": False, "zarr_label": int(label_value), "expected_zarr_label": int(graph_tid), "area": int(area), "gt_overlap_pixels": 0, "reason": "outside_gt_region", "centroid_label_distance": centroid_dist})
                    continue
            sc = SingleCellStatic(
                timeframe=t,
                img_dataset=img_dataset,
                contour=np.asarray(contour, dtype=float),
                bbox=np.asarray(bbox, dtype=int),
                id=f"{source_name}_{graph_tid}_{t}",
                feature_dict={"ultrack_label_area": float(area), "ultrack_graph_track_id": float(graph_tid)},
                meta={
                    SingleCellMetaKeyManager.MASK_LABEL: int(label_value),
                    "ultrack_graph_track_id": int(graph_tid),
                    "ultrack_node_id": int(row["id"]) if "id" in row and pd.notna(row["id"]) else None,
                    "source_zarr_label": int(label_value),
                    "centroid_zarr_label": int(centroid_label),
                    "centroid_label_distance": int(centroid_dist),
                },
            )
            track_to_cells[graph_tid][t] = sc
            rows.append({"track_id": graph_tid, "time": t, "converted": True, "kept": True, "zarr_label": int(label_value), "expected_zarr_label": int(graph_tid), "area": int(area), "gt_overlap_pixels": overlap_pixels, "reason": "", "centroid_label_distance": centroid_dist})

    sctc = SingleCellTrajectoryCollection()
    for track_id in sorted(track_to_cells):
        cells = dict(sorted(track_to_cells[track_id].items()))
        if not cells:
            continue
        traj = SingleCellTrajectory(
            track_id=int(track_id),
            timeframe_to_single_cell=cells,
            img_dataset=img_dataset,
            meta={
                SingleCellTrajectory.META_MOTHER_IDS: [],
                SingleCellTrajectory.META_DAUGHTER_IDS: [],
                "source": source_name,
                "source_zarr_path": str(resolve_path(zarr_path)),
                "trajectory_id_source": "ultrack_to_tracks_layer_track_id",
            },
        )
        sctc.add_trajectory(traj)

    lineage_stats = add_lineage_links(sctc, graph)
    return sctc, pd.DataFrame(rows), lineage_stats


def backup_if_needed(path: Path) -> None:
    if not path.exists():
        return
    backup = path.with_suffix(path.suffix + ".bak_before_tracksdf_lineage_rebuild")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[backup] {backup}")


def write_outputs(sctc: SingleCellTrajectoryCollection, rows: pd.DataFrame, lineage_stats: dict, output_path: Path, dataset_dir_name: str, description: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup_if_needed(output_path)
    dataset_dir = output_path.parent / dataset_dir_name
    sctc.write_json(str(output_path), dataset_json_dir=dataset_dir)
    rows.to_csv(output_path.with_suffix(".tracksdf_rebuild_cells.csv"), index=False)
    readme = output_path.with_suffix(".tracksdf_rebuild.README.md")
    readme.write_text(
        "\n".join(
            [
                f"# {description}",
                "",
                "This SCTC was rebuilt from Ultrack `to_tracks_layer` tracks_df IDs, not zarr label IDs.",
                "Mother-daughter links were assigned using the Ultrack child->parent graph after rebuilding the SCTC with matching track IDs.",
                f"Output: `{output_path}`",
                f"Trajectories: {len(sctc)}",
                f"Cells: {len(sctc.get_all_scs())}",
                f"Lineage stats: `{json.dumps(lineage_stats, sort_keys=True)}`",
            ]
        )
        + "\n"
    )
    print(f"[write] {output_path}")
    print(f"[stats] {lineage_stats}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-path", default=str(GT_PATH))
    parser.add_argument("--ultrack-zarr-path", default=str(ULTRACK_EXP4_ZARR_PATH))
    parser.add_argument("--tracks-df", default=str(DEFAULT_TRACKS_DF))
    parser.add_argument("--graph-json", default=str(DEFAULT_GRAPH_JSON))
    parser.add_argument("--full-output", default=str(DEFAULT_FULL_OUTPUT))
    parser.add_argument("--gt-region-output", default=str(DEFAULT_GT_REGION_OUTPUT))
    parser.add_argument("--only", choices=["both", "full", "gt-region"], default="both")
    parser.add_argument("--skip-source-audit", action="store_true", help="Do not use unless deliberately debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = resolve_path(Path(args.gt_path))
    zarr_path = resolve_path(Path(args.ultrack_zarr_path))
    tracks_df_path = resolve_path(Path(args.tracks_df))
    graph_path = resolve_path(Path(args.graph_json))
    tracks_df = pd.read_csv(tracks_df_path)
    graph = {int(k): int(v) for k, v in json.loads(graph_path.read_text()).items()}
    if not args.skip_source_audit:
        audit_tracks_df_zarr_pair(tracks_df, zarr_path)

    gt_sctc = load_sctc(gt_path)
    img_dataset = next(iter(gt_sctc.track_id_to_trajectory.values())).img_dataset
    if img_dataset is None:
        raise ValueError("GT SCTC does not contain image dataset metadata.")

    if args.only in {"both", "full"}:
        full_sctc, full_rows, full_lineage = build_sctc_from_tracks_df(
            tracks_df, graph, zarr_path, img_dataset, gt_union_masks=None, source_name="ultrack_tracksdf_full"
        )
        write_outputs(
            full_sctc,
            full_rows,
            full_lineage,
            Path(args.full_output),
            "datasets_ultrack_full",
            "Full Ultrack SCTC rebuilt from tracks_df IDs",
        )

    if args.only in {"both", "gt-region"}:
        labels = ZarrV3Labels(zarr_path)
        gt_ids = relation_ids_from_gt(gt_sctc)
        gt_cell_df = build_gt_cell_table(gt_sctc, gt_ids)
        gt_union_masks = build_gt_union_masks(gt_sctc, gt_cell_df, tuple(labels.shape[1:]))
        gt_sctc_rebuilt, gt_rows, gt_lineage = build_sctc_from_tracks_df(
            tracks_df, graph, zarr_path, img_dataset, gt_union_masks=gt_union_masks, source_name="ultrack_tracksdf_gt_region_only"
        )
        write_outputs(
            gt_sctc_rebuilt,
            gt_rows,
            gt_lineage,
            Path(args.gt_region_output),
            "datasets_gt_region_only",
            "GT-region Ultrack SCTC rebuilt from tracks_df IDs",
        )


if __name__ == "__main__":
    main()
