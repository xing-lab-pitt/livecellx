#!/usr/bin/env python
"""Trajectory statistics for process2 tracking outputs.

This script compares three process2 tracking outputs:

* SORT: trajectories before CSNet/LiveCellX correction.
* LiveCellX: corrected trajectories. By default this uses the
  TimeSformer-lineage-corrected SCTC if present, then falls back to the
  after-CSNet SCTC.
* Ultrack: focused GT-region-only trajectory collection by default; a label zarr
  path can still be supplied for older whole-label checks.

Outputs are written to:
    revision_code/results_process2_csnet/traj_statistics
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-livecellx")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process2_ctc_style_tracking_evaluation import (
    GT_PATH,
    LATEST_RESULTS_DIR,
    MatchConfig,
    RESULTS_DIR,
    SCRIPT_DIR,
    SORT_AFTER_PATH,
    SORT_BEFORE_FALLBACK_PATH,
    SORT_BEFORE_PATH,
    TIMESFORMER_CORRECTED_AFTER_PATH,
    ULTRACK_ZARR_PATH,
    is_sctc_json_path,
    ZarrV3Labels,
    bbox_area,
    bbox_intersection,
    bbox_intersects,
    build_gt_cell_table,
    build_time_index,
    load_sctc,
    match_sctc_condition,
    match_ultrack_condition,
    progress_iter,
    relation_ids_from_gt,
    resolve_path,
)


TRAJ_STATS_DIR = LATEST_RESULTS_DIR / "traj_statistics"

METHOD_ORDER = ["SORT", "LiveCellX", "Ultrack"]
METHOD_COLORS = {
    "SORT": "#3F6DB5",
    "LiveCellX": "#F28E2B",
    "Ultrack": "#009E73",
}
SHORT_LABELS = ["SORT", "LiveCellX", "Ultrack"]


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


def save_figure(fig: plt.Figure, out_base: Path, dpi: int = 450) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def polygon_area_from_contour(contour) -> float:
    arr = np.asarray(contour, dtype=float)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return np.nan
    y = arr[:, 0]
    x = arr[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return area if area > 0 else np.nan


def robust_area_from_sc(sc, exact_mask_area: bool = False) -> float:
    if sc.bbox is None:
        return np.nan
    bbox = np.asarray(sc.bbox, dtype=int)
    bbox_area = float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))
    if exact_mask_area:
        try:
            mask = sc.get_contour_mask(bbox=bbox).astype(bool)
            area = float(mask.sum())
            return area if area > 0 else bbox_area
        except Exception:
            return bbox_area
    try:
        area = polygon_area_from_contour(getattr(sc, "contour", None))
        return area if area > 0 else bbox_area
    except Exception:
        return bbox_area


def morphology_from_sc(sc) -> dict:
    """Return mask-shape features from a SingleCell contour mask."""
    try:
        from skimage.measure import label, regionprops

        mask = sc.get_contour_mask(crop=True, dtype=bool)
        props = regionprops(label(mask.astype(np.uint8)))
        if not props:
            raise ValueError("empty contour mask")
        prop = props[0]
        return {
            "perimeter": float(prop.perimeter),
            "eccentricity": float(prop.eccentricity),
            "solidity": float(prop.solidity),
        }
    except Exception:
        return {"perimeter": np.nan, "eccentricity": np.nan, "solidity": np.nan}


def relation_counts_from_traj(traj) -> Tuple[int, int]:
    mothers = set()
    daughters = set()
    try:
        mothers.update(int(t.track_id) for t in traj.mother_trajectories)
    except Exception:
        pass
    try:
        daughters.update(int(t.track_id) for t in traj.daughter_trajectories)
    except Exception:
        pass
    meta = getattr(traj, "meta", {}) or {}
    for key in ("mother_trajectory_ids", "mother_ids"):
        for val in meta.get(key, []) or []:
            if val is not None:
                mothers.add(int(val))
    for key in ("daughter_trajectory_ids", "daughter_ids"):
        for val in meta.get(key, []) or []:
            if val is not None:
                daughters.add(int(val))
    return len(mothers), len(daughters)


def sctc_to_cell_features(method: str, path: Path, exact_mask_area: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sctc = load_sctc(resolve_path(path))
    cell_rows: List[dict] = []
    relation_rows: List[dict] = []
    for tid, traj in progress_iter(list(sctc), total=len(sctc), desc=f"{method}: SCTC trajectories"):
        tid = int(tid)
        n_mothers, n_daughters = relation_counts_from_traj(traj)
        relation_rows.append(
            {
                "method": method,
                "track_id": tid,
                "n_mothers": n_mothers,
                "n_daughters": n_daughters,
                "has_mother": n_mothers > 0,
                "has_daughter": n_daughters > 0,
                "is_branching_mother": n_daughters >= 2,
            }
        )
        for t in sorted(int(x) for x in traj.times):
            sc = traj.get_sc(t)
            if sc.bbox is None:
                continue
            bbox = np.asarray(sc.bbox, dtype=float)
            height = float(max(0.0, bbox[2] - bbox[0]))
            width = float(max(0.0, bbox[3] - bbox[1]))
            area = robust_area_from_sc(sc, exact_mask_area=exact_mask_area)
            morph = morphology_from_sc(sc)
            cell_rows.append(
                {
                    "method": method,
                    "track_id": tid,
                    "time": int(t),
                    "area": area,
                    "bbox_area": float(width * height),
                    "height": height,
                    "width": width,
                    "aspect_ratio": float(width / height) if height > 0 else np.nan,
                    "perimeter": morph["perimeter"],
                    "eccentricity": morph["eccentricity"],
                    "solidity": morph["solidity"],
                    "centroid_y": float((bbox[0] + bbox[2]) / 2.0),
                    "centroid_x": float((bbox[1] + bbox[3]) / 2.0),
                }
            )
    return pd.DataFrame(cell_rows), pd.DataFrame(relation_rows)


def ultrack_to_cell_features(method: str, zarr_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from skimage.measure import regionprops_table

    labels = ZarrV3Labels(resolve_path(zarr_path))
    rows: List[pd.DataFrame] = []
    for t in progress_iter(range(labels.shape[0]), total=labels.shape[0], desc=f"{method}: zarr frames"):
        frame = labels.read_frame(int(t))
        props = regionprops_table(
            frame,
            properties=("label", "area", "bbox", "centroid", "perimeter", "eccentricity", "solidity"),
        )
        if len(props.get("label", [])) == 0:
            continue
        df = pd.DataFrame(props)
        df = df.rename(
            columns={
                "label": "track_id",
                "bbox-0": "bbox_y0",
                "bbox-1": "bbox_x0",
                "bbox-2": "bbox_y1",
                "bbox-3": "bbox_x1",
                "centroid-0": "centroid_y",
                "centroid-1": "centroid_x",
            }
        )
        df["method"] = method
        df["time"] = int(t)
        df["height"] = (df["bbox_y1"] - df["bbox_y0"]).astype(float)
        df["width"] = (df["bbox_x1"] - df["bbox_x0"]).astype(float)
        df["bbox_area"] = df["height"] * df["width"]
        df["aspect_ratio"] = np.where(df["height"] > 0, df["width"] / df["height"], np.nan)
        keep = [
            "method",
            "track_id",
            "time",
            "area",
            "bbox_area",
            "height",
            "width",
            "aspect_ratio",
            "perimeter",
            "eccentricity",
            "solidity",
            "centroid_y",
            "centroid_x",
        ]
        rows.append(df[keep])
    cell_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    relation_df = pd.DataFrame(
        {
            "method": method,
            "track_id": sorted(cell_df["track_id"].dropna().astype(int).unique()) if len(cell_df) else [],
        }
    )
    if len(relation_df):
        relation_df["n_mothers"] = 0
        relation_df["n_daughters"] = 0
        relation_df["has_mother"] = False
        relation_df["has_daughter"] = False
        relation_df["is_branching_mother"] = False
    return cell_df, relation_df


def sctc_relation_df_for_tids(
    method: str,
    sctc,
    tids: Iterable[int],
) -> pd.DataFrame:
    rows = []
    for tid in sorted({int(x) for x in tids if pd.notna(x)}):
        if tid not in sctc.track_id_to_trajectory:
            continue
        n_mothers, n_daughters = relation_counts_from_traj(sctc.get_trajectory(tid))
        rows.append(
            {
                "method": method,
                "track_id": tid,
                "n_mothers": n_mothers,
                "n_daughters": n_daughters,
                "has_mother": n_mothers > 0,
                "has_daughter": n_daughters > 0,
                "is_branching_mother": n_daughters >= 2,
            }
        )
    return pd.DataFrame(rows)


def empty_relation_df(method: str, tids: Iterable[int]) -> pd.DataFrame:
    tids = sorted({int(x) for x in tids if pd.notna(x)})
    relation_df = pd.DataFrame({"method": method, "track_id": tids})
    if len(relation_df):
        relation_df["n_mothers"] = 0
        relation_df["n_daughters"] = 0
        relation_df["has_mother"] = False
        relation_df["has_daughter"] = False
        relation_df["is_branching_mother"] = False
    return relation_df


def gt_restricted_sctc_cell_features(
    method: str,
    target_sctc,
    match_df: pd.DataFrame,
    exact_mask_area: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    matched_tids = []
    records = match_df.sort_values(["gt_tid", "time"]).to_dict("records")
    for row in progress_iter(records, total=len(records), desc=f"{method}: GT-region SCTC features"):
        t = int(row["time"])
        target_tid = row.get("target_tid")
        matched = bool(row.get("matched", False)) and pd.notna(target_tid)
        feat = {
            "method": method,
            "analysis_scope": "gt_region",
            "track_id": np.nan,
            "time": t,
            "area": np.nan,
            "bbox_area": np.nan,
            "height": np.nan,
            "width": np.nan,
            "aspect_ratio": np.nan,
            "perimeter": np.nan,
            "eccentricity": np.nan,
            "solidity": np.nan,
            "centroid_y": np.nan,
            "centroid_x": np.nan,
            "matched": bool(row.get("matched", False)),
            "iou": float(row.get("iou", np.nan)),
            "gt_tid": int(row["gt_tid"]),
            "gt_time": t,
            "gt_start": int(row.get("gt_start", np.nan)),
            "gt_end": int(row.get("gt_end", np.nan)),
            "gt_length": int(row.get("gt_length", np.nan)),
        }
        if matched:
            target_tid = int(target_tid)
            try:
                traj = target_sctc.get_trajectory(target_tid)
                if t in set(int(x) for x in traj.times):
                    sc = traj.get_sc(t)
                    if sc.bbox is not None:
                        bbox = np.asarray(sc.bbox, dtype=float)
                        height = float(max(0.0, bbox[2] - bbox[0]))
                        width = float(max(0.0, bbox[3] - bbox[1]))
                        morph = morphology_from_sc(sc)
                        feat.update(
                            {
                                "track_id": target_tid,
                                "area": robust_area_from_sc(sc, exact_mask_area=exact_mask_area),
                                "bbox_area": float(width * height),
                                "height": height,
                                "width": width,
                                "aspect_ratio": float(width / height) if height > 0 else np.nan,
                                "perimeter": morph["perimeter"],
                                "eccentricity": morph["eccentricity"],
                                "solidity": morph["solidity"],
                                "centroid_y": float((bbox[0] + bbox[2]) / 2.0),
                                "centroid_x": float((bbox[1] + bbox[3]) / 2.0),
                            }
                        )
                        matched_tids.append(target_tid)
            except Exception:
                pass
        rows.append(feat)
    return pd.DataFrame(rows), sctc_relation_df_for_tids(method, target_sctc, matched_tids)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = bbox_intersection(a, b)
    if inter is None:
        return 0.0
    inter_area = bbox_area(inter)
    union = bbox_area(a) + bbox_area(b) - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def match_sctc_condition_fast_bbox(
    condition: str,
    target_sctc,
    gt_sctc,
    gt_cell_df: pd.DataFrame,
    cfg: MatchConfig,
) -> pd.DataFrame:
    allowed_times = set(int(x) for x in gt_cell_df["time"].unique())
    target_by_time = build_time_index(target_sctc, allowed_times)
    rows = []
    records = gt_cell_df.to_dict("records")
    print(f"[match-fast] {condition}: {len(records)} GT cells over {len(allowed_times)} frames")
    for row in progress_iter(records, total=len(records), desc=f"match-fast {condition}"):
        gt_tid = int(row["gt_tid"])
        t = int(row["time"])
        gt_sc = gt_sctc.get_trajectory(gt_tid).get_sc(t)
        best_tid = None
        best_iou = 0.0
        if gt_sc.bbox is not None:
            gt_bbox = np.asarray(gt_sc.bbox, dtype=float)
            for cand in target_by_time.get(t, []):
                if not bbox_intersects(gt_bbox, cand.bbox):
                    continue
                iou = bbox_iou(gt_bbox, cand.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = int(cand.tid)
        rows.append(
            {
                **row,
                "condition": condition,
                "target_tid": best_tid,
                "iou": float(best_iou),
                "matched": bool(best_tid is not None and best_iou >= cfg.iou_threshold),
            }
        )
    return pd.DataFrame(rows)


def match_ultrack_condition_fast_bbox(
    condition: str,
    zarr_path: Path,
    gt_sctc,
    gt_cell_df: pd.DataFrame,
    cfg: MatchConfig,
) -> pd.DataFrame:
    labels = ZarrV3Labels(resolve_path(zarr_path))
    rows = []
    records = gt_cell_df.sort_values("time").to_dict("records")
    print(f"[match-fast] {condition}: {len(records)} GT cells against {zarr_path}")
    for row in progress_iter(records, total=len(records), desc=f"match-fast {condition}"):
        gt_tid = int(row["gt_tid"])
        t = int(row["time"])
        gt_sc = gt_sctc.get_trajectory(gt_tid).get_sc(t)
        best_label = None
        best_iou = 0.0
        if gt_sc.bbox is not None and 0 <= t < labels.shape[0]:
            bbox = np.asarray(gt_sc.bbox, dtype=int)
            y0 = max(0, int(bbox[0]))
            x0 = max(0, int(bbox[1]))
            y1 = min(labels.shape[1], int(bbox[2]))
            x1 = min(labels.shape[2], int(bbox[3]))
            if y1 > y0 and x1 > x0:
                frame = labels.read_frame(t)
                crop = frame[y0:y1, x0:x1]
                inside = crop[crop > 0]
                if len(inside):
                    cand_labels, intersections = np.unique(inside, return_counts=True)
                    areas = labels.label_areas(t)
                    gt_area = float((y1 - y0) * (x1 - x0))
                    for label_value, inter in zip(cand_labels, intersections):
                        pred_area = float(areas.get(int(label_value), 0))
                        union = gt_area + pred_area - float(inter)
                        iou = float(inter / union) if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_label = int(label_value)
        rows.append(
            {
                **row,
                "condition": condition,
                "target_tid": best_label,
                "iou": float(best_iou),
                "matched": bool(best_label is not None and best_iou >= cfg.iou_threshold),
            }
        )
    return pd.DataFrame(rows)


def gt_restricted_ultrack_cell_features(
    method: str,
    zarr_path: Path,
    match_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from skimage.measure import regionprops_table

    labels = ZarrV3Labels(resolve_path(zarr_path))
    feature_parts: List[pd.DataFrame] = []
    for t in progress_iter(
        sorted(int(x) for x in match_df["time"].unique()),
        total=match_df["time"].nunique(),
        desc=f"{method}: GT-region zarr features",
    ):
        frame = labels.read_frame(int(t))
        props = regionprops_table(
            frame,
            properties=("label", "area", "bbox", "centroid", "perimeter", "eccentricity", "solidity"),
        )
        if len(props.get("label", [])) == 0:
            continue
        df = pd.DataFrame(props).rename(
            columns={
                "label": "track_id",
                "bbox-0": "bbox_y0",
                "bbox-1": "bbox_x0",
                "bbox-2": "bbox_y1",
                "bbox-3": "bbox_x1",
                "centroid-0": "centroid_y",
                "centroid-1": "centroid_x",
            }
        )
        df["time"] = int(t)
        df["height"] = (df["bbox_y1"] - df["bbox_y0"]).astype(float)
        df["width"] = (df["bbox_x1"] - df["bbox_x0"]).astype(float)
        df["bbox_area"] = df["height"] * df["width"]
        df["aspect_ratio"] = np.where(df["height"] > 0, df["width"] / df["height"], np.nan)
        df["target_tid_key"] = df["track_id"].astype("Int64")
        feature_parts.append(
            df[
                [
                    "time",
                    "target_tid_key",
                    "area",
                    "bbox_area",
                    "height",
                    "width",
                    "aspect_ratio",
                    "perimeter",
                    "eccentricity",
                    "solidity",
                    "centroid_y",
                    "centroid_x",
                ]
            ]
        )

    feature_df = pd.concat(feature_parts, ignore_index=True) if feature_parts else pd.DataFrame()
    base = match_df.copy()
    base["target_tid_key"] = base["target_tid"].astype("Int64")
    base = base.rename(columns={"time": "gt_time"})
    base["time"] = base["gt_time"].astype(int)
    if len(feature_df):
        merged = base.merge(feature_df, on=["time", "target_tid_key"], how="left")
    else:
        merged = base.copy()
        for col in [
            "area",
            "bbox_area",
            "height",
            "width",
            "aspect_ratio",
            "perimeter",
            "eccentricity",
            "solidity",
            "centroid_y",
            "centroid_x",
        ]:
            merged[col] = np.nan
    merged["method"] = method
    merged["analysis_scope"] = "gt_region"
    merged["track_id"] = merged["target_tid"]
    keep = [
        "method",
        "analysis_scope",
        "track_id",
        "time",
        "area",
        "bbox_area",
        "height",
        "width",
        "aspect_ratio",
        "perimeter",
        "eccentricity",
        "solidity",
        "centroid_y",
        "centroid_x",
        "matched",
        "iou",
        "gt_tid",
        "gt_time",
        "gt_start",
        "gt_end",
        "gt_length",
    ]
    cell_df = merged[keep].copy()
    return cell_df, empty_relation_df(method, cell_df.loc[cell_df["matched"], "track_id"].dropna())


def build_track_statistics(cell_df: pd.DataFrame, relation_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    if cell_df.empty:
        return pd.DataFrame()
    work_df = cell_df.copy()
    if "matched" in work_df.columns:
        work_df = work_df[work_df["matched"].astype(bool) & work_df["track_id"].notna()].copy()
    if work_df.empty:
        return pd.DataFrame()
    for (method, tid), group in progress_iter(
        list(work_df.groupby(["method", "track_id"])),
        total=work_df.groupby(["method", "track_id"]).ngroups,
        desc="trajectory statistics",
    ):
        group = group.sort_values("time")
        if group["time"].duplicated().any():
            sort_cols = ["time"]
            ascending = [True]
            if "iou" in group.columns:
                sort_cols.append("iou")
                ascending.append(False)
            group = group.sort_values(sort_cols, ascending=ascending).drop_duplicates("time", keep="first")
            group = group.sort_values("time")
        times = group["time"].astype(int).to_numpy()
        length = int(len(group))
        start = int(times.min())
        end = int(times.max())
        span = int(end - start + 1)
        dt = np.diff(times)
        consecutive = dt == 1
        gaps = dt[dt > 1] - 1
        y = group["centroid_y"].astype(float).to_numpy()
        x = group["centroid_x"].astype(float).to_numpy()
        steps = np.sqrt(np.diff(y) ** 2 + np.diff(x) ** 2)
        consec_steps = steps[consecutive] if len(steps) else np.asarray([])
        area = group["area"].astype(float).to_numpy()
        log_area = np.log1p(area)
        area_delta = np.abs(np.diff(log_area))
        consec_area_delta = area_delta[consecutive] if len(area_delta) else np.asarray([])
        path_length = float(np.nansum(consec_steps)) if len(consec_steps) else 0.0
        net = float(math.hypot(y[-1] - y[0], x[-1] - x[0])) if length > 1 else 0.0
        rows.append(
            {
                "method": method,
                "track_id": int(tid),
                "length": length,
                "start_time": start,
                "end_time": end,
                "span": span,
                "vacancy_rate": float((span - length) / span) if span > 0 else np.nan,
                "presence_fraction": float(length / span) if span > 0 else np.nan,
                "n_gaps": int(len(gaps)),
                "max_gap": int(gaps.max()) if len(gaps) else 0,
                "mean_area": float(np.nanmean(area)),
                "median_area": float(np.nanmedian(area)),
                "area_cv": float(np.nanstd(area) / np.nanmean(area)) if np.nanmean(area) > 0 else np.nan,
                "mean_area_abs_log_change": float(np.nanmean(consec_area_delta)) if len(consec_area_delta) else np.nan,
                "max_area_abs_log_change": float(np.nanmax(consec_area_delta)) if len(consec_area_delta) else np.nan,
                "mean_step_displacement": float(np.nanmean(consec_steps)) if len(consec_steps) else np.nan,
                "median_step_displacement": float(np.nanmedian(consec_steps)) if len(consec_steps) else np.nan,
                "max_step_displacement": float(np.nanmax(consec_steps)) if len(consec_steps) else np.nan,
                "path_length": path_length,
                "net_displacement": net,
                "straightness": float(net / path_length) if path_length > 0 else np.nan,
            }
        )
    stats = pd.DataFrame(rows)
    if relation_df is not None and len(relation_df):
        stats = stats.merge(relation_df, on=["method", "track_id"], how="left")
    for col in ["n_mothers", "n_daughters"]:
        if col in stats.columns:
            stats[col] = stats[col].fillna(0).astype(int)
    for col in ["has_mother", "has_daughter", "is_branching_mother"]:
        if col in stats.columns:
            stats[col] = stats[col].fillna(False).astype(bool)
    return stats


def build_summary(cell_df: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        c = cell_df[cell_df["method"] == method]
        t = track_df[track_df["method"] == method]
        n_region_cells = int(len(c))
        if "matched" in c.columns:
            n_cells = int(c["matched"].astype(bool).sum())
            detection_rate = float(c["matched"].astype(bool).mean()) if len(c) else np.nan
            mean_iou_matched = float(c.loc[c["matched"].astype(bool), "iou"].mean()) if c["matched"].any() else np.nan
        else:
            n_cells = int(len(c))
            detection_rate = np.nan
            mean_iou_matched = np.nan
        if t.empty:
            rows.append(
                {
                    "method": method,
                    "analysis_scope": str(c["analysis_scope"].iloc[0]) if "analysis_scope" in c.columns and len(c) else "gt_region",
                    "n_gt_region_cells": n_region_cells,
                    "n_cells": n_cells,
                    "detection_rate": detection_rate,
                    "mean_iou_matched": mean_iou_matched,
                    "mean_thresholded_iou_all_gt": np.nan,
                    "n_trajectories": 0,
                    "median_length": np.nan,
                    "mean_length": np.nan,
                    "median_span": np.nan,
                    "median_vacancy_rate": np.nan,
                    "tracks_with_gaps_rate": np.nan,
                    "short_track_rate_len_le_3": np.nan,
                    "complete_presence_rate": np.nan,
                    "median_area_cv": np.nan,
                    "median_step_displacement": np.nan,
                    "branching_mother_count": 0,
                    "daughter_track_count": 0,
                }
            )
            continue
        rows.append(
            {
                "method": method,
                "analysis_scope": str(c["analysis_scope"].iloc[0]) if "analysis_scope" in c.columns and len(c) else "whole_movie",
                "n_gt_region_cells": n_region_cells,
                "n_cells": n_cells,
                "detection_rate": detection_rate,
                "mean_iou_matched": mean_iou_matched,
                "mean_thresholded_iou_all_gt": float(
                    np.where(c["matched"].astype(bool), c["iou"].astype(float), 0.0).mean()
                )
                if len(c) and "matched" in c.columns and "iou" in c.columns
                else np.nan,
                "n_trajectories": int(len(t)),
                "median_length": float(t["length"].median()),
                "mean_length": float(t["length"].mean()),
                "median_span": float(t["span"].median()),
                "median_vacancy_rate": float(t["vacancy_rate"].median()),
                "tracks_with_gaps_rate": float((t["n_gaps"] > 0).mean()),
                "short_track_rate_len_le_3": float((t["length"] <= 3).mean()),
                "complete_presence_rate": float((t["vacancy_rate"] == 0).mean()),
                "median_area_cv": float(t["area_cv"].median()),
                "median_step_displacement": float(t["mean_step_displacement"].median()),
                "branching_mother_count": int(t.get("is_branching_mother", pd.Series(False, index=t.index)).sum()),
                "daughter_track_count": int(t.get("has_mother", pd.Series(False, index=t.index)).sum()),
            }
        )
    return pd.DataFrame(rows)


def boxplot_by_method(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    log_y: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    data = [
        df.loc[(df["method"] == method) & np.isfinite(df[value_col]), value_col].to_numpy()
        for method in METHOD_ORDER
    ]
    bp = ax.boxplot(
        data,
        tick_labels=SHORT_LABELS,
        showfliers=False,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#222222", "linewidth": 1.2},
        whiskerprops={"color": "#333333", "linewidth": 0.8},
        capprops={"color": "#333333", "linewidth": 0.8},
    )
    for box, method in zip(bp["boxes"], METHOD_ORDER):
        box.set_facecolor(METHOD_COLORS[method])
        box.set_alpha(0.82)
        box.set_edgecolor("#333333")
        box.set_linewidth(0.8)
    ax.set_title(title, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(False)


def plot_summary_metric(
    summary_df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    title: str,
    ylabel: str,
    output_stem: str,
    pct: bool = False,
) -> None:
    vals = []
    for method in METHOD_ORDER:
        sub = summary_df[summary_df["method"] == method]
        vals.append(float(sub[metric].iloc[0]) if len(sub) and metric in sub.columns else np.nan)
    vals_plot = [value * 100 if pct and pd.notna(value) else value for value in vals]
    x = np.arange(len(METHOD_ORDER))
    fig, ax = plt.subplots(figsize=(3.35, 2.75), constrained_layout=True)
    ax.bar(
        x,
        vals_plot,
        color=[METHOD_COLORS[method] for method in METHOD_ORDER],
        edgecolor="white",
        linewidth=0.6,
        width=0.62,
    )
    finite_vals = [float(value) for value in vals_plot if pd.notna(value)]
    ymax = max(finite_vals) if finite_vals else 1.0
    label_offset = 0.025 * ymax if ymax > 0 else 0.04
    for xpos, value in zip(x, vals_plot):
        if pd.notna(value):
            label = f"{value:.1f}" if pct else f"{value:.3f}"
            ax.text(xpos, value + label_offset, label, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS)
    ax.set_title(title, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(False)
    if pct:
        ax.set_ylim(0, max(100.0, ymax + 4.0))
    save_figure(fig, out_dir / output_stem)


def plot_summary_figures(summary_df: pd.DataFrame, out_dir: Path) -> None:
    # Primary GT-referenced evidence.
    plot_summary_metric(
        summary_df,
        out_dir,
        "detection_rate",
        "Recovery of annotated GT cells",
        "Recovered GT cells (%)",
        "fig_traj_gt_cell_recovery",
        pct=True,
    )
    plot_summary_metric(
        summary_df,
        out_dir,
        "mean_thresholded_iou_all_gt",
        "GT-region mask agreement",
        "Mean IoU (misses = 0)",
        "fig_traj_gt_mask_agreement",
    )

    # These are separate diagnostics because predicted-track statistics can
    # reward an incorrect mother-daughter identity merge.
    diagnostic_specs = [
        ("n_trajectories", "Track IDs covering GT cells", "No. track IDs", "fig_traj_track_id_count_diagnostic", False),
        ("median_length", "Median predicted-track length", "Frames", "fig_traj_median_track_length_diagnostic", False),
        ("tracks_with_gaps_rate", "Predicted tracks with gaps", "Tracks (%)", "fig_traj_tracks_with_gaps_diagnostic", True),
        ("short_track_rate_len_le_3", "Very short predicted tracks", "Tracks (%)", "fig_traj_very_short_track_rate", True),
    ]
    for metric, title, ylabel, stem, pct in diagnostic_specs:
        plot_summary_metric(summary_df, out_dir, metric, title, ylabel, stem, pct=pct)


def plot_distribution_figures(track_df: pd.DataFrame, out_dir: Path) -> None:
    panel_specs = [
        ("length", "Predicted-track length", "Frames", True, None, "fig_traj_track_length_distribution_diagnostic"),
        ("span", "Predicted-track span", "Frames", True, None, "fig_traj_track_span_distribution_diagnostic"),
        ("n_gaps", "Internal missing intervals", "No. intervals", False, None, "fig_traj_internal_gap_distribution_diagnostic"),
        ("max_gap", "Largest internal gap", "Frames", False, None, "fig_traj_largest_gap_distribution_diagnostic"),
        ("mean_step_displacement", "Frame-to-frame displacement", "Pixels/frame", True, None, "fig_traj_step_displacement_distribution_diagnostic"),
        ("mean_area_abs_log_change", "Frame-to-frame area change", "|delta log(area)|", True, None, "fig_traj_area_jump_distribution_diagnostic"),
    ]
    for value_col, title, ylabel, log_y, ylim, output_stem in panel_specs:
        fig, ax = plt.subplots(figsize=(3.35, 2.75), constrained_layout=True)
        boxplot_by_method(ax, track_df, value_col, title, ylabel, log_y=log_y, ylim=ylim)
        save_figure(fig, out_dir / output_stem)


def plot_time_distribution_figures(track_df: pd.DataFrame, out_dir: Path) -> None:
    bins = np.arange(
        int(min(track_df["start_time"].min(), track_df["end_time"].min())),
        int(max(track_df["start_time"].max(), track_df["end_time"].max())) + 2,
        10,
    )
    specs = [
        ("start_time", "Predicted-track start times", "fig_traj_start_time_distribution_diagnostic"),
        ("end_time", "Predicted-track end times", "fig_traj_end_time_distribution_diagnostic"),
    ]
    for value_col, title, output_stem in specs:
        fig, ax = plt.subplots(figsize=(3.6, 2.75), constrained_layout=True)
        for method in METHOD_ORDER:
            sub = track_df[track_df["method"] == method]
            ax.hist(
                sub[value_col],
                bins=bins,
                histtype="step",
                linewidth=1.5,
                color=METHOD_COLORS[method],
                label=method,
            )
        ax.set_title(title, fontweight="bold", loc="left")
        ax.set_xlabel("Frame")
        ax.set_ylabel("No. trajectories")
        ax.grid(False)
        ax.legend(frameon=False, loc="upper right")
        save_figure(fig, out_dir / output_stem)


def remove_legacy_combined_figures(out_dir: Path) -> None:
    for stem in (
        "fig_traj_summary_statistics",
        "fig_traj_distribution_panels",
        "fig_traj_start_end_time_distributions",
    ):
        for suffix in (".pdf", ".png", ".svg"):
            (out_dir / f"{stem}{suffix}").unlink(missing_ok=True)

def write_readme(out_dir: Path, summary_df: pd.DataFrame, sources: Dict[str, str]) -> None:
    lines = [
        "# Process2 trajectory statistics",
        "",
        "This folder contains trajectory statistics for SORT, LiveCellX, and Ultrack tracking outputs.",
        "",
        "Default scope: annotated GT region only, restricted to frames 0-99 unless overridden. Each row is anchored to a cell from the hand-annotated process2 GT lineage trajectories; cells outside those GT annotations are ignored.",
        "",
        "## Source files",
        "",
    ]
    if sources.get("GT"):
        lines.append(f"- GT annotations: `{sources.get('GT')}`")
    for method in METHOD_ORDER:
        lines.append(f"- {method}: `{sources.get(method, '')}`")
    lines += [
        "",
        "## Output tables",
        "",
        "- `process2_cell_features.csv`: per-cell/per-label measurements used by downstream analysis.",
        "- `process2_track_statistics.csv`: one row per trajectory or Ultrack label ID.",
        "- `process2_trajectory_summary.csv`: method-level summary table.",
        "",
        "SCTC cell area is measured from the SingleCell contour polygon by default for speed. Use `--exact-mask-area` if you want to rasterize every SCTC contour mask, which is much slower.",
        "",
        "## Figure-use hierarchy",
        "",
        "Every metric is exported as an independent figure. The old multi-panel summary, distribution, and start/end figures are removed when this script runs.",
        "",
        "### Recommended rebuttal evidence",
        "",
        "- `fig_traj_gt_cell_recovery`: percentage of annotated GT cells recovered at the same frame with IoU at or above the matching threshold. This directly tests whether a method loses fewer annotated cells. It supports the statement that LiveCellX improves cell recovery in the annotated region, but it does not test trajectory identity or division links.",
        "- `fig_traj_gt_mask_agreement`: mean same-frame mask IoU across all annotated GT cells, assigning zero to cells below the matching threshold. This combines recovery and mask agreement. It is useful as segmentation support, but it does not establish correct mother-daughter identity.",
        "",
        "### Secondary evidence",
        "",
        "- `fig_traj_very_short_track_rate`: fraction of predicted track IDs represented for three frames or fewer in the annotated region. This can indicate fragmentation, but only as supporting evidence because an incorrect long identity merge can artificially lower this rate.",
        "",
        "### Diagnostics; not recommended as headline rebuttal evidence",
        "",
        "- `fig_traj_track_id_count_diagnostic`: number of predicted IDs covering matched GT cells. Fewer IDs are not necessarily better because a method can incorrectly merge mother and daughter identities.",
        "- `fig_traj_median_track_length_diagnostic`: median length of predicted IDs. Longer is not necessarily better for the same identity-merging reason.",
        "- `fig_traj_tracks_with_gaps_diagnostic`: fraction of predicted IDs with internal missing intervals. This describes continuity within predicted IDs but does not verify that the ID follows the correct biological cell.",
        "- `fig_traj_track_length_distribution_diagnostic` and `fig_traj_track_span_distribution_diagnostic`: distributions underlying predicted-ID length and span; subject to identity-merge and observation-window confounding.",
        "- `fig_traj_internal_gap_distribution_diagnostic` and `fig_traj_largest_gap_distribution_diagnostic`: predicted-ID gap distributions; useful for quality control, not lineage correctness.",
        "- `fig_traj_step_displacement_distribution_diagnostic`: motion distribution. Lower displacement is not intrinsically better without a GT motion reference.",
        "- `fig_traj_area_jump_distribution_diagnostic`: temporal mask-area-change distribution. Lower values may indicate smoother segmentation, but biological size changes and identity errors can also affect it. In the current data this metric does not show a clear LiveCellX improvement, so it should not be used to claim one.",
        "- `fig_traj_start_time_distribution_diagnostic` and `fig_traj_end_time_distribution_diagnostic`: start/end distributions. They are strongly affected by the 0-99 frame boundary and track splitting/merging and add little to the rebuttal.",
        "",
        "The absolute recovered-cell count is not plotted because every method uses the same GT denominator, making it a duplicate of GT-cell recovery. Branching and identity correctness must be assessed with the separate GT-based trajectory-level metrics rather than these predicted-ID diagnostics.",
        "",
        "## Term definitions",
        "",
        "- `annotated GT region`: the subset of process2 defined by the hand-annotated GT lineage trajectories. All default statistics here are restricted to these GT cells and their matched method cells; unannotated movie regions are ignored.",
        "- `GT cell`: one manually annotated cell mask at one frame in the GT trajectory collection.",
        "- `matched` or `recovered` cell: a method cell/label at the same frame that overlaps the GT cell with IoU >= the matching threshold.",
        "- `IoU`: intersection-over-union overlap between the GT cell and the method cell. For these descriptive statistics, the default matching uses fast bounding-box IoU; run with `--exact-match-iou` for rasterized mask IoU.",
        "- `method`: the tracking result being evaluated: SORT, LiveCellX, or Ultrack.",
        "- `analysis_scope`: whether rows are from the annotated GT region (`gt_region`) or the optional whole-movie mode (`whole_movie`). Default is `gt_region`.",
        "- `n_gt_region_cells`: number of annotated GT cells used as evaluation anchors. This should be the same for all methods.",
        "- `n_cells`: number of GT-region cells recovered by the method.",
        "- `detection_rate` or `GT-cell recovery`: `n_cells / n_gt_region_cells`; higher means fewer annotated GT cells are missed.",
        "- `mean_iou_matched`: average IoU among recovered GT-region cells; higher means the recovered masks align better with the GT masks. Because missed cells are excluded, this is diagnostic and is not plotted as the main mask-agreement result.",
        "- `mean_thresholded_iou_all_gt`: mean same-frame IoU across every annotated GT cell after assigning zero to cells below the matching threshold. This is the value plotted as GT-region mask agreement.",
        "- `track ID` or `trajectory ID`: the label/trajectory identifier assigned by a tracking method.",
        "- `n_trajectories`: number of unique method track IDs needed to cover the matched GT-region cells. A larger value can indicate fragmentation when the same GT lineage is split across many method IDs.",
        "- `length`: number of matched GT-region frames present for one method track ID.",
        "- `span`: frame range covered by one method track ID, calculated as last matched frame minus first matched frame plus one.",
        "- `median_length` and `mean_length`: median or mean number of matched GT-region frames per method track ID.",
        "- `median_span`: median frame span per method track ID.",
        "- `gap`: a missing internal frame between the first and last matched frame of the same method track ID.",
        "- `n_gaps`: number of internal missing intervals in one method track ID.",
        "- `max_gap`: length of the largest internal missing interval for one method track ID.",
        "- `vacancy_rate`: fraction of frames missing inside a track span, calculated as `(span - length) / span`.",
        "- `median_vacancy_rate`: median vacancy rate across method track IDs.",
        "- `tracks_with_gaps_rate`: fraction of method track IDs with at least one internal gap. Lower means better temporal continuity.",
        "- `complete_presence_rate`: fraction of method track IDs with no internal missing frames.",
        "- `short_track_rate_len_le_3`: fraction of method track IDs with length <= 3 frames. Lower means fewer very short fragments.",
        "- `area`: cell size measured from the method cell contour or label region.",
        "- `area_cv`: coefficient of variation of cell area along one method track ID. Higher values can indicate unstable segmentation over time.",
        "- `median_area_cv`: median area variability across method track IDs.",
        "- `mean_area_abs_log_change`: mean absolute frame-to-frame change in log cell area. Higher values indicate stronger temporal area jumps.",
        "- `step displacement`: frame-to-frame centroid movement of one track ID, in pixels per frame.",
        "- `median_step_displacement`: median frame-to-frame centroid movement across method track IDs.",
        "- `branching_mother_count` and `daughter_track_count`: explicit mother/daughter relation counts stored in SCTC metadata. These columns are kept in the table for SCTC inspection, but they are not used for the default fair comparison because lineage metadata is not stored equivalently across all methods.",
        "",
        "## Current summary",
        "",
    ]
    if not summary_df.empty:
        lines.append("```")
        lines.append(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("```")
    (out_dir / "README_process2_trajectory_statistics.md").write_text("\n".join(lines))


def default_after_path() -> Path:
    return SORT_AFTER_PATH


def build_whole_movie_feature_tables(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    before_path = resolve_path(Path(args.before_path), Path(args.before_fallback_path))
    after_path = resolve_path(Path(args.after_path))
    ultrack_path = resolve_path(Path(args.ultrack_zarr_path))
    sources = {
        "SORT": str(before_path),
        "LiveCellX": str(after_path),
        "Ultrack": str(ultrack_path),
    }

    cell_parts = []
    relation_parts = []
    before_cells, before_rel = sctc_to_cell_features(
        "SORT",
        before_path,
        exact_mask_area=args.exact_mask_area,
    )
    after_cells, after_rel = sctc_to_cell_features(
        "LiveCellX",
        after_path,
        exact_mask_area=args.exact_mask_area,
    )
    if is_sctc_json_path(ultrack_path):
        ultrack_cells, ultrack_rel = sctc_to_cell_features(
            "Ultrack",
            ultrack_path,
            exact_mask_area=args.exact_mask_area,
        )
    else:
        ultrack_cells, ultrack_rel = ultrack_to_cell_features("Ultrack", ultrack_path)
    cell_parts.extend([before_cells, after_cells, ultrack_cells])
    relation_parts.extend([before_rel, after_rel, ultrack_rel])
    cell_df = pd.concat(cell_parts, ignore_index=True)
    cell_df["analysis_scope"] = "whole_movie"
    relation_df = pd.concat(relation_parts, ignore_index=True)
    track_df = build_track_statistics(cell_df, relation_df)
    if len(track_df):
        track_df["analysis_scope"] = "whole_movie"
    summary_df = build_summary(cell_df, track_df)
    summary_df.attrs["sources"] = sources
    return cell_df, track_df, summary_df


def build_gt_region_feature_tables(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt_path = resolve_path(Path(args.gt_path))
    before_path = resolve_path(Path(args.before_path), Path(args.before_fallback_path))
    after_path = resolve_path(Path(args.after_path))
    ultrack_path = resolve_path(Path(args.ultrack_zarr_path))
    cfg = MatchConfig(iou_threshold=float(args.iou_threshold))
    sources = {
        "GT": str(gt_path),
        "SORT": str(before_path),
        "LiveCellX": str(after_path),
        "Ultrack": str(ultrack_path),
    }

    gt_sctc = load_sctc(gt_path)
    gt_ids = relation_ids_from_gt(gt_sctc)
    gt_cell_df = build_gt_cell_table(gt_sctc, gt_ids, min_frame=int(args.min_frame), max_frame=args.max_frame)
    if gt_cell_df.empty:
        raise ValueError("No annotated GT cells found. Cannot build GT-region statistics.")
    print(
        f"[gt-region] restricting statistics to {len(gt_cell_df)} cells "
        f"from {len(gt_ids)} annotated GT lineage trajectories in frames {int(args.min_frame)}-{args.max_frame}"
    )

    before_sctc = load_sctc(before_path)
    sctc_matcher = match_sctc_condition if args.exact_match_iou else match_sctc_condition_fast_bbox
    ultrack_matcher = match_ultrack_condition if args.exact_match_iou else match_ultrack_condition_fast_bbox

    before_match = sctc_matcher("SORT", before_sctc, gt_sctc, gt_cell_df, cfg)
    before_cells, before_rel = gt_restricted_sctc_cell_features(
        "SORT",
        before_sctc,
        before_match,
        exact_mask_area=args.exact_mask_area,
    )
    del before_sctc

    after_sctc = load_sctc(after_path)
    after_match = sctc_matcher("LiveCellX", after_sctc, gt_sctc, gt_cell_df, cfg)
    after_cells, after_rel = gt_restricted_sctc_cell_features(
        "LiveCellX",
        after_sctc,
        after_match,
        exact_mask_area=args.exact_mask_area,
    )
    del after_sctc

    if is_sctc_json_path(ultrack_path):
        ultrack_sctc = load_sctc(ultrack_path)
        ultrack_match = sctc_matcher("Ultrack", ultrack_sctc, gt_sctc, gt_cell_df, cfg)
        ultrack_cells, ultrack_rel = gt_restricted_sctc_cell_features(
            "Ultrack",
            ultrack_sctc,
            ultrack_match,
            exact_mask_area=args.exact_mask_area,
        )
        del ultrack_sctc
    else:
        ultrack_match = ultrack_matcher("Ultrack", ultrack_path, gt_sctc, gt_cell_df, cfg)
        ultrack_cells, ultrack_rel = gt_restricted_ultrack_cell_features("Ultrack", ultrack_path, ultrack_match)

    cell_df = pd.concat([before_cells, after_cells, ultrack_cells], ignore_index=True)
    relation_df = pd.concat([before_rel, after_rel, ultrack_rel], ignore_index=True)
    track_df = build_track_statistics(cell_df, relation_df)
    if len(track_df):
        track_df["analysis_scope"] = "gt_region"
    summary_df = build_summary(cell_df, track_df)
    summary_df.attrs["sources"] = sources
    return cell_df, track_df, summary_df


def build_feature_tables(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if getattr(args, "whole_movie", False):
        return build_whole_movie_feature_tables(args)
    return build_gt_region_feature_tables(args)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_csv = out_dir / "process2_cell_features.csv"
    track_csv = out_dir / "process2_track_statistics.csv"
    summary_csv = out_dir / "process2_trajectory_summary.csv"

    expected_scope = "whole_movie" if args.whole_movie else "gt_region"
    can_reuse = args.reuse_csv and cell_csv.exists() and track_csv.exists() and summary_csv.exists()
    if can_reuse:
        try:
            scope_probe = pd.read_csv(cell_csv, nrows=5)
            found_scope = set(scope_probe.get("analysis_scope", pd.Series(dtype=str)).dropna().astype(str))
            can_reuse = bool(found_scope) and found_scope == {expected_scope}
            if can_reuse and expected_scope == "gt_region" and "time" in scope_probe.columns:
                full_time_probe = pd.read_csv(cell_csv, usecols=["time"])
                if len(full_time_probe):
                    tmin = int(full_time_probe["time"].min())
                    tmax = int(full_time_probe["time"].max())
                    if tmin < int(args.min_frame) or tmax > int(args.max_frame):
                        print(
                            f"[reuse] existing CSV frame range {tmin}-{tmax} is outside "
                            f"requested {int(args.min_frame)}-{args.max_frame}; recomputing"
                        )
                        can_reuse = False
            if not can_reuse:
                print(f"[reuse] existing CSV scope/frame range is not {expected_scope}; recomputing")
        except Exception:
            can_reuse = False

    if can_reuse:
        print(f"[reuse] reading existing CSV files from {out_dir}")
        cell_df = pd.read_csv(cell_csv)
        track_df = pd.read_csv(track_csv)
        summary_df = pd.read_csv(summary_csv)
        sources = {
            "GT": str(resolve_path(Path(args.gt_path))),
            "SORT": str(resolve_path(Path(args.before_path), Path(args.before_fallback_path))),
            "LiveCellX": str(resolve_path(Path(args.after_path))),
            "Ultrack": str(resolve_path(Path(args.ultrack_zarr_path))),
        }
    else:
        cell_df, track_df, summary_df = build_feature_tables(args)
        sources = summary_df.attrs.get("sources", {})
        cell_df.to_csv(cell_csv, index=False)
        track_df.to_csv(track_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)

    if "mean_thresholded_iou_all_gt" not in summary_df.columns:
        values = {}
        for method in METHOD_ORDER:
            sub = cell_df[cell_df["method"] == method]
            values[method] = float(
                np.where(sub["matched"].astype(bool), sub["iou"].astype(float), 0.0).mean()
            ) if len(sub) else np.nan
        summary_df["mean_thresholded_iou_all_gt"] = summary_df["method"].map(values)
        summary_df.to_csv(summary_csv, index=False)

    remove_legacy_combined_figures(out_dir)
    plot_summary_figures(summary_df, out_dir)
    plot_distribution_figures(track_df, out_dir)
    plot_time_distribution_figures(track_df, out_dir)
    write_readme(out_dir, summary_df, sources)
    print(f"[done] saved trajectory statistics to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-path", default=str(GT_PATH))
    parser.add_argument("--before-path", default=str(SORT_BEFORE_PATH))
    parser.add_argument("--before-fallback-path", default=str(SORT_BEFORE_FALLBACK_PATH))
    parser.add_argument("--after-path", default=str(default_after_path()))
    parser.add_argument(
        "--ultrack-zarr-path",
        default=str(ULTRACK_ZARR_PATH),
        help=(
            "Ultrack input. The default is the focused GT-region-only Ultrack SCTC JSON; "
            "a zarr label folder is also accepted."
        ),
    )
    parser.add_argument("--output-dir", default=str(TRAJ_STATS_DIR))
    parser.add_argument("--min-frame", type=int, default=0, help="First frame included in GT-region statistics.")
    parser.add_argument("--max-frame", type=int, default=99, help="Last frame included in GT-region statistics; default keeps frames 0-99.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--reuse-csv",
        action="store_true",
        help="Regenerate figures from existing CSVs in --output-dir when possible.",
    )
    parser.add_argument(
        "--exact-mask-area",
        action="store_true",
        help="Rasterize SCTC contours for exact mask area. Default uses fast contour polygon area.",
    )
    parser.add_argument(
        "--exact-match-iou",
        action="store_true",
        help="Use exact rasterized mask IoU for GT-region matching. Default uses fast bbox IoU for descriptive statistics.",
    )
    parser.add_argument(
        "--whole-movie",
        action="store_true",
        help="Use the old whole-movie scope instead of restricting statistics to annotated GT cells.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
