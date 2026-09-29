#!/usr/bin/env python
"""Downstream analysis figures for process2 tracking outputs.

The script uses the same three tracking outputs as
`process2_trajectory_statistics.py`, then summarizes downstream consequences:

* feature distributions,
* temporal feature stability,
* trajectory fragmentation / continuity proxies.

An exploratory cell-feature embedding can still be generated with
`--make-embedding`, but it is not part of the default rebuttal figure set
because the key claim is tracking-result quality in the annotated GT region.

Outputs are written to:
    revision_code/results_process2_csnet/downstream_analysis
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-livecellx")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from livecellx.trajectory.contour.contour_class import Contour

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process2_trajectory_statistics import (
    METHOD_COLORS,
    METHOD_ORDER,
    SHORT_LABELS,
    TRAJ_STATS_DIR,
    build_feature_tables,
    match_sctc_condition_fast_bbox,
    morphology_from_sc,
    relation_counts_from_traj,
    robust_area_from_sc,
    save_figure,
)
from process2_ctc_style_tracking_evaluation import (
    FINAL_TRAJ_COLLECTION_DIR,
    GT_PATH,
    LATEST_RESULTS_DIR,
    MatchConfig,
    RESULTS_DIR,
    ULTRACK_ZARR_PATH,
    best_sctc_match,
    build_time_index,
    is_sctc_json_path,
    lineage_edges_from_gt,
    load_sctc,
    relation_ids_from_gt,
    resolve_path,
)


DOWNSTREAM_DIR = LATEST_RESULTS_DIR / "downstream_analysis"
TRAJECTORY_RESULTS_SUBDIR = "trajectory_based_results"
NO_GT_LIVECELLX_TRACK_ID = 440
NO_GT_SORT_TRACK_ID = 413
NO_GT_CASE_FOLDER = "no_gt_livecellx_0440_sort_0413"
DOWNSTREAM_SORT_BEFORE_FULL_PATH = (
    FINAL_TRAJ_COLLECTION_DIR / "sctc-final-2026-0610_before_livecellx_full.json"
)
DOWNSTREAM_SORT_AFTER_FULL_PATH = (
    FINAL_TRAJ_COLLECTION_DIR
    / "sctc-final-2026-06-02_before_correction_timesformer_lineage_corrected_full.json"
)
DOWNSTREAM_ULTRACK_FULL_PATH = (
    FINAL_TRAJ_COLLECTION_DIR / "ultrack_from_cellpose_process2_beforecsnet_exp4_full.json"
)

# Preserve the established downstream-analysis palette independently of benchmark figures.
METHOD_COLORS = {"SORT": "#56B4E9", "LiveCellX": "#F2C94C", "Ultrack": "#E45756"}

GT_LINE_COLOR = "#009E73"
TRAJECTORY_METHOD_ORDER = ["GT", "SORT", "LiveCellX"]
TRAJECTORY_METHOD_ORDER_WITH_ULTRACK = ["GT", "SORT", "LiveCellX", "Ultrack"]
TRAJECTORY_METHOD_COLORS = {
    "GT": GT_LINE_COLOR,
    "SORT": "#56B4E9",
    "LiveCellX": "#F2C94C",
    "Ultrack": "#E45756",
}
TRAJECTORY_LINE_STYLES = {
    "GT": {"linestyle": "-", "marker": "o", "linewidth": 2.80, "markersize": 1.70, "alpha": 0.30, "zorder": 1},
    "SORT": {"linestyle": "--", "marker": "s", "linewidth": 1.05, "markersize": 1.45, "alpha": 0.40, "zorder": 2},
    "LiveCellX": {"linestyle": "-", "marker": "^", "linewidth": 1.20, "markersize": 1.50, "alpha": 0.40, "zorder": 3},
    "Ultrack": {"linestyle": "-.", "marker": "D", "linewidth": 1.05, "markersize": 1.35, "alpha": 0.40, "zorder": 4},
}
TRAJECTORY_SHAPE_MARKERS = {
    "GT": "o",
    "SORT": "s",
    "LiveCellX": "^",
    "Ultrack": "D",
}
DOWNSTREAM_METHOD_ORDER_NO_ULTRACK = ["SORT", "LiveCellX"]
DOWNSTREAM_SHORT_LABELS = {
    "SORT": "SORT",
    "LiveCellX": "LiveCellX",
    "Ultrack": "Ultrack",
}


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


def ensure_feature_tables(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats_dir = Path(args.traj_statistics_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    cell_csv = stats_dir / "process2_cell_features.csv"
    track_csv = stats_dir / "process2_track_statistics.csv"
    summary_csv = stats_dir / "process2_trajectory_summary.csv"
    source_json = stats_dir / "process2_downstream_feature_sources.json"
    expected_scope = "whole_movie" if args.whole_movie else "gt_region"
    expected_sources = {
        "analysis_scope": expected_scope,
        "gt_path": str(resolve_path(Path(args.gt_path))),
        "before_path": str(resolve_path(Path(args.before_path), Path(args.before_fallback_path))),
        "after_path": str(resolve_path(Path(args.after_path))),
        "ultrack_zarr_path": str(resolve_path(Path(args.ultrack_zarr_path))),
        "iou_threshold": float(args.iou_threshold),
        "min_frame": int(args.min_frame),
        "max_frame": None if args.max_frame is None else int(args.max_frame),
        "exact_match_iou": bool(args.exact_match_iou),
        "exact_mask_area": bool(args.exact_mask_area),
    }
    required_cell_cols = {"area", "perimeter", "eccentricity", "solidity", "aspect_ratio"}
    can_reuse = cell_csv.exists() and track_csv.exists() and summary_csv.exists() and not args.recompute_features
    if can_reuse:
        try:
            if not source_json.exists():
                print("[reuse] existing feature CSVs have no source signature; recomputing")
                can_reuse = False
            else:
                found_sources = json.loads(source_json.read_text())
                if found_sources != expected_sources:
                    print("[reuse] existing feature CSVs were built from different inputs; recomputing")
                    can_reuse = False
            scope_probe = pd.read_csv(cell_csv, nrows=5)
            found_scope = set(scope_probe.get("analysis_scope", pd.Series(dtype=str)).dropna().astype(str))
            can_reuse = can_reuse and bool(found_scope) and found_scope == {expected_scope}
            found_methods = set(pd.read_csv(cell_csv, usecols=["method"])["method"].dropna().astype(str))
            if found_methods != set(METHOD_ORDER):
                print(f"[reuse] existing feature CSV uses method labels {sorted(found_methods)}; recomputing")
                can_reuse = False
            missing_cols = required_cell_cols.difference(scope_probe.columns)
            if missing_cols:
                print(f"[reuse] existing cell CSV is missing {sorted(missing_cols)}; recomputing")
                can_reuse = False
            if not can_reuse:
                print(f"[reuse] existing trajectory CSV scope is not {expected_scope}; recomputing")
        except Exception:
            can_reuse = False
    if can_reuse:
        print(f"[reuse] loading trajectory feature CSVs from {stats_dir}")
        return pd.read_csv(cell_csv), pd.read_csv(track_csv), pd.read_csv(summary_csv)

    print("[build] trajectory feature CSVs are missing or --recompute-features was used")
    cell_df, track_df, summary_df = build_feature_tables(args)
    cell_df.to_csv(cell_csv, index=False)
    track_df.to_csv(track_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    source_json.write_text(json.dumps(expected_sources, indent=2, sort_keys=True))
    return cell_df, track_df, summary_df


def finite_series(df: pd.DataFrame, col: str, method: str) -> np.ndarray:
    vals = df.loc[(df["method"] == method) & np.isfinite(df[col]), col].to_numpy(dtype=float)
    return vals


def boxplot_by_method(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    log_y: bool = False,
    methods: Sequence[str] = METHOD_ORDER,
) -> None:
    data = [finite_series(df, value_col, method) for method in methods]
    bp = ax.boxplot(
        data,
        tick_labels=[DOWNSTREAM_SHORT_LABELS.get(method, method) for method in methods],
        showfliers=False,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#222222", "linewidth": 1.2},
        whiskerprops={"color": "#333333", "linewidth": 0.8},
        capprops={"color": "#333333", "linewidth": 0.8},
    )
    for box, method in zip(bp["boxes"], methods):
        box.set_facecolor(METHOD_COLORS[method])
        box.set_alpha(0.82)
        box.set_edgecolor("#333333")
        box.set_linewidth(0.8)
    ax.set_title(title, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(False)


def downstream_summary(track_df: pd.DataFrame) -> pd.DataFrame:
    area_jump_threshold = float(track_df["mean_area_abs_log_change"].quantile(0.90))
    displacement_threshold = float(track_df["mean_step_displacement"].quantile(0.90))
    rows = []
    for method in METHOD_ORDER:
        sub = track_df[track_df["method"] == method].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "method": method,
                "n_trajectories": int(len(sub)),
                "median_length": float(sub["length"].median()),
                "short_track_rate_len_le_3": float((sub["length"] <= 3).mean()),
                "fragmented_track_rate": float((sub["vacancy_rate"] > 0).mean()),
                "high_vacancy_rate_gt_10pct": float((sub["vacancy_rate"] > 0.10).mean()),
                "median_area_cv": float(sub["area_cv"].median()),
                "median_area_abs_log_change": float(sub["mean_area_abs_log_change"].median()),
                "high_area_jump_rate_top10pct": float((sub["mean_area_abs_log_change"] >= area_jump_threshold).mean()),
                "median_step_displacement": float(sub["mean_step_displacement"].median()),
                "high_displacement_rate_top10pct": float((sub["mean_step_displacement"] >= displacement_threshold).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_downstream_quality_summary(
    summary_df: pd.DataFrame,
    out_dir: Path,
    methods: Sequence[str] = METHOD_ORDER,
    filename_suffix: str = "",
) -> None:
    specs = [
        ("short_track_rate_len_le_3", "Very short matched tracks", "Tracks (%)"),
        ("fragmented_track_rate", "Tracks with gaps", "Tracks (%)"),
        ("high_vacancy_rate_gt_10pct", "Tracks missing >10% of span", "Tracks (%)"),
        ("high_area_jump_rate_top10pct", "High area-jump tracks", "Tracks (%)"),
        ("high_displacement_rate_top10pct", "High displacement tracks", "Tracks (%)"),
        ("median_length", "Median GT-region track length", "Frames"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 4.8), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, specs):
        vals = []
        for method in methods:
            sub = summary_df[summary_df["method"] == method]
            vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
        if ylabel.endswith("(%)"):
            vals = [v * 100 if pd.notna(v) else np.nan for v in vals]
        x = np.arange(len(methods))
        ax.bar(x, vals, color=[METHOD_COLORS[m] for m in methods], edgecolor="white", linewidth=0.6)
        ymax = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
        for xpos, val in zip(x, vals):
            if pd.notna(val):
                ax.text(xpos, val + ymax * 0.025, f"{val:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([DOWNSTREAM_SHORT_LABELS.get(method, method) for method in methods])
        ax.set_title(title, fontweight="bold", loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(False)
    fig.suptitle("Downstream artifact burden in annotated GT regions", fontweight="bold", fontsize=11)
    save_figure(fig, out_dir / f"fig_downstream_quality_summary{filename_suffix}")


def plot_feature_distributions(
    cell_df: pd.DataFrame,
    track_df: pd.DataFrame,
    out_dir: Path,
    methods: Sequence[str] = METHOD_ORDER,
    filename_suffix: str = "",
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.2), constrained_layout=True)
    specs = [
        (cell_df, "area", "Cell area", "Pixels", True),
        (cell_df, "aspect_ratio", "Cell shape ratio", "Width / height", True),
        (track_df, "area_cv", "Trajectory area variability", "CV", True),
        (track_df, "mean_area_abs_log_change", "Temporal area jump", "|delta log(area)|", True),
        (track_df, "mean_step_displacement", "Step displacement", "Pixels/frame", True),
        (track_df, "straightness", "Trajectory straightness", "Net / path", False),
    ]
    for ax, (df, col, title, ylabel, log_y) in zip(axes.flat, specs):
        boxplot_by_method(ax, df, col, title, ylabel, log_y=log_y, methods=methods)
    fig.suptitle("Downstream feature distributions", fontweight="bold", fontsize=11)
    save_figure(fig, out_dir / f"fig_downstream_feature_distributions{filename_suffix}")


def plot_temporal_metric_trend(
    cell_df: pd.DataFrame,
    out_dir: Path,
    value_col: str,
    title: str,
    ylabel: str,
    filename: str,
    transform=None,
) -> None:
    tmp = cell_df.copy()
    if "analysis_scope" in tmp.columns:
        tmp = tmp[tmp["analysis_scope"].astype(str).eq("gt_region")].copy()
    if "matched" in tmp.columns:
        tmp = tmp[tmp["matched"].fillna(False).astype(bool)].copy()
    tmp = tmp[np.isfinite(tmp[value_col])].copy()
    if transform is not None:
        tmp["_plot_value"] = transform(tmp[value_col].astype(float))
    else:
        tmp["_plot_value"] = tmp[value_col].astype(float)
    grouped = (
        tmp.groupby(["method", "time"])
        .agg(
            median_value=("_plot_value", "median"),
            n_cells=("track_id", "count"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
    for method in METHOD_ORDER:
        sub = grouped[grouped["method"] == method]
        if sub.empty:
            continue
        ax.plot(
            sub["time"],
            sub["median_value"],
            color=METHOD_COLORS[method],
            linewidth=1.5,
            label=method,
        )
    ax.set_title(title, fontweight="bold", loc="left")
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    save_figure(fig, out_dir / filename)


def plot_temporal_feature_trends(cell_df: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        ("area", "Median cell area", "log(1 + area)", "fig_downstream_temporal_area", np.log1p),
        ("perimeter", "Median cell perimeter", "Perimeter (pixels)", "fig_downstream_temporal_perimeter", None),
        ("eccentricity", "Median eccentricity", "Eccentricity", "fig_downstream_temporal_eccentricity", None),
        ("solidity", "Median solidity", "Solidity", "fig_downstream_temporal_solidity", None),
        ("aspect_ratio", "Median aspect ratio", "Width / height", "fig_downstream_temporal_aspect_ratio", None),
    ]
    for value_col, title, ylabel, filename, transform in specs:
        if value_col not in cell_df.columns:
            print(f"[skip] missing downstream feature column: {value_col}")
            continue
        plot_temporal_metric_trend(
            cell_df,
            out_dir,
            value_col=value_col,
            title=title,
            ylabel=ylabel,
            filename=filename,
            transform=transform,
        )


def clean_trajectory_results_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.pdf", "*.png", "*.svg", "*.csv", "*.md"):
        for path in out_dir.glob(pattern):
            path.unlink()
    for path in out_dir.iterdir():
        if path.is_dir() and (path.name.startswith("gt_traj_") or path.name == NO_GT_CASE_FOLDER):
            shutil.rmtree(path)


def feature_row_from_sc(method: str, track_id: int, time: int, sc, exact_mask_area: bool) -> dict:
    bbox = np.asarray(sc.bbox, dtype=float) if sc.bbox is not None else np.asarray([np.nan] * 4)
    height = float(max(0.0, bbox[2] - bbox[0])) if np.isfinite(bbox).all() else np.nan
    width = float(max(0.0, bbox[3] - bbox[1])) if np.isfinite(bbox).all() else np.nan
    morph = morphology_from_sc(sc)
    return {
        "method": method,
        "track_id": int(track_id),
        "time": int(time),
        "available": True,
        "area": robust_area_from_sc(sc, exact_mask_area=exact_mask_area),
        "bbox_area": float(width * height) if np.isfinite(width) and np.isfinite(height) else np.nan,
        "height": height,
        "width": width,
        "aspect_ratio": float(width / height) if np.isfinite(width) and np.isfinite(height) and height > 0 else np.nan,
        "perimeter": morph["perimeter"],
        "eccentricity": morph["eccentricity"],
        "solidity": morph["solidity"],
        "centroid_y": float((bbox[0] + bbox[2]) / 2.0) if np.isfinite(bbox).all() else np.nan,
        "centroid_x": float((bbox[1] + bbox[3]) / 2.0) if np.isfinite(bbox).all() else np.nan,
    }


def empty_feature_row(method: str, track_id: Optional[int], time: int) -> dict:
    return {
        "method": method,
        "track_id": int(track_id) if track_id is not None and pd.notna(track_id) else np.nan,
        "time": int(time),
        "available": False,
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
    }


def extract_sctc_series_for_gt_span(
    method: str,
    sctc,
    track_id: Optional[int],
    start_t: int,
    end_t: int,
    exact_mask_area: bool,
) -> List[dict]:
    rows: List[dict] = []
    if track_id is None or pd.isna(track_id) or int(track_id) not in sctc.track_id_to_trajectory:
        return [empty_feature_row(method, None, t) for t in range(int(start_t), int(end_t) + 1)]
    track_id = int(track_id)
    traj = sctc.get_trajectory(track_id)
    times = set(int(t) for t in traj.times)
    for t in range(int(start_t), int(end_t) + 1):
        if t not in times:
            rows.append(empty_feature_row(method, track_id, t))
            continue
        sc = traj.get_sc(t)
        if sc is None or sc.bbox is None:
            rows.append(empty_feature_row(method, track_id, t))
            continue
        rows.append(feature_row_from_sc(method, track_id, t, sc, exact_mask_area=exact_mask_area))
    return rows


def build_gt_trajectory_anchor_table(gt_sctc, gt_ids: Sequence[int], min_frame: int = 0, max_frame: Optional[int] = 99) -> pd.DataFrame:
    rows: List[dict] = []
    for gt_tid in sorted(int(x) for x in gt_ids):
        if gt_tid not in gt_sctc.track_id_to_trajectory:
            continue
        traj = gt_sctc.get_trajectory(gt_tid)
        times = sorted(int(t) for t in traj.times if int(t) >= int(min_frame) and (max_frame is None or int(t) <= int(max_frame)))
        if not times:
            continue
        n_mothers, n_daughters = relation_counts_from_traj(traj)
        if n_daughters > 0:
            role = "mother"
            anchor_policy = "mother_first_gt_cell"
            anchor_time = int(times[0])
        elif n_mothers > 0:
            role = "daughter"
            anchor_policy = "daughter_first_gt_cell"
            anchor_time = int(times[0])
        else:
            role = "annotated_traj"
            anchor_policy = "single_traj_first_gt_cell"
            anchor_time = int(times[0])
        rows.append(
            {
                "gt_tid": int(gt_tid),
                "time": int(anchor_time),
                "gt_start": int(times[0]),
                "gt_end": int(times[-1]),
                "gt_length": int(len(times)),
                "_gt_times": tuple(int(t) for t in times),
                "gt_role": role,
                "anchor_policy": anchor_policy,
                "gt_n_mothers": int(n_mothers),
                "gt_n_daughters": int(n_daughters),
            }
        )
    return pd.DataFrame(rows)


def target_tid_from_anchor(row: pd.Series) -> Optional[int]:
    if row is None or row.empty:
        return None
    if pd.isna(row.get("target_tid", np.nan)):
        return None
    return int(row["target_tid"])


def safe_get_gt_sc(gt_sctc, gt_tid: int, t: int):
    if int(gt_tid) not in gt_sctc.track_id_to_trajectory:
        return None
    traj = gt_sctc.get_trajectory(int(gt_tid))
    return getattr(traj, "timeframe_to_single_cell", {}).get(int(t))


def expand_anchor_rows_for_lookup(anchor_df: pd.DataFrame, search_radius: int) -> pd.DataFrame:
    """Create exact-anchor plus nearby-frame lookup rows for trajectory ID selection.

    The exact anchor is always rank 0.  For trajectory-feature plots, every GT
    trajectory is anchored from its earliest annotated cell.  Nearby-frame
    fallback therefore searches forward within that GT trajectory if the exact
    first frame was dropped by tracking.
    """
    rows: List[dict] = []
    search_radius = int(search_radius)
    for _, row in anchor_df.iterrows():
        anchor_time = int(row["time"])
        gt_start = int(row["gt_start"])
        gt_end = int(row["gt_end"])
        raw_times = row.get("_gt_times", None)
        if isinstance(raw_times, (list, tuple, np.ndarray, pd.Series)):
            available_times = sorted({int(t) for t in raw_times})
        else:
            available_times = list(range(gt_start, gt_end + 1))
        times = [t for t in available_times if anchor_time <= t <= gt_end]
        times = sorted(times)
        if search_radius >= 0:
            times = [t for t in times if abs(t - anchor_time) <= search_radius]
        for rank, t in enumerate(times):
            new_row = row.to_dict()
            new_row["time"] = int(t)
            new_row["exact_anchor_time"] = int(anchor_time)
            new_row["anchor_lookup_rank"] = int(rank)
            new_row["anchor_lookup_offset"] = int(abs(t - anchor_time))
            rows.append(new_row)
    return pd.DataFrame(rows)


def select_anchor_matches(match_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows: List[dict] = []
    for gt_tid, sub in match_df.groupby("gt_tid"):
        sub = sub.copy()
        sub["has_target"] = sub["target_tid"].notna()
        found = sub[sub["has_target"]].sort_values(["anchor_lookup_rank", "iou"], ascending=[True, False])
        if found.empty:
            selected = sub.sort_values(["anchor_lookup_rank", "iou"], ascending=[True, False]).iloc[0]
        else:
            selected = found.iloc[0]
        target_tid = target_tid_from_anchor(selected)
        rows.append(
            {
                "gt_tid": int(gt_tid),
                f"{prefix}_track_id": target_tid,
                f"{prefix}_anchor_iou": float(selected.get("iou", np.nan)),
                f"{prefix}_anchor_time_used": int(selected.get("time")),
                f"{prefix}_anchor_lookup_offset": int(selected.get("anchor_lookup_offset", 0)),
                f"{prefix}_anchor_found": target_tid is not None,
                f"{prefix}_anchor_pass_iou_threshold": bool(selected.get("matched", False)),
            }
        )
    return pd.DataFrame(rows)


def select_local_overlap_anchor_matches(
    condition: str,
    target_sctc,
    gt_sctc,
    lookup_df: pd.DataFrame,
    cfg: MatchConfig,
    prefix: str,
) -> pd.DataFrame:
    """Select a target trajectory ID from the GT anchor region.

    This is intentionally trajectory-ID lookup, not filtering by an IoU
    threshold.  For each GT trajectory anchor we search the exact frame first,
    then the permitted forward/backward frames.  At each real GT frame, the
    target cell with the largest overlap with the GT cell region is selected.
    The IoU threshold is only recorded as a diagnostic pass/fail column.
    """
    rows: List[dict] = []
    lookup_df = lookup_df.copy()
    allowed_times = set(int(t) for t in lookup_df["time"].dropna().unique())
    target_by_time = build_time_index(target_sctc, allowed_times)
    area_cache: Dict[int, float] = {}

    for gt_tid, sub in lookup_df.groupby("gt_tid"):
        selected = None
        sub = sub.sort_values(["anchor_lookup_rank", "time"])
        for _, row in sub.iterrows():
            t = int(row["time"])
            gt_sc = safe_get_gt_sc(gt_sctc, int(gt_tid), t)
            if gt_sc is None:
                continue
            target_tid, iou = best_sctc_match(gt_sc, target_by_time.get(t, []), cfg, area_cache)
            selected = {
                "time": t,
                "target_tid": target_tid,
                "iou": float(iou),
                "matched": bool(target_tid is not None and iou >= cfg.iou_threshold),
                "anchor_lookup_offset": int(row.get("anchor_lookup_offset", 0)),
                "anchor_lookup_rank": int(row.get("anchor_lookup_rank", 0)),
            }
            if target_tid is not None:
                break
        if selected is None:
            first = sub.iloc[0]
            selected = {
                "time": int(first["time"]),
                "target_tid": None,
                "iou": 0.0,
                "matched": False,
                "anchor_lookup_offset": int(first.get("anchor_lookup_offset", 0)),
                "anchor_lookup_rank": int(first.get("anchor_lookup_rank", 0)),
            }
        target_tid = selected["target_tid"]
        rows.append(
            {
                "gt_tid": int(gt_tid),
                f"{prefix}_track_id": None if target_tid is None else int(target_tid),
                f"{prefix}_anchor_iou": float(selected["iou"]),
                f"{prefix}_anchor_time_used": int(selected["time"]),
                f"{prefix}_anchor_lookup_offset": int(selected["anchor_lookup_offset"]),
                f"{prefix}_anchor_found": target_tid is not None,
                f"{prefix}_anchor_pass_iou_threshold": bool(selected["matched"]),
            }
        )
    print(f"[trajectory-anchor] {condition}: selected local target IDs for {len(rows)} GT trajectories")
    return pd.DataFrame(rows)


def build_gt_anchored_trajectory_mapping(
    gt_sctc,
    before_sctc,
    after_sctc,
    ultrack_sctc,
    anchor_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    cfg = MatchConfig(iou_threshold=float(args.trajectory_anchor_min_iou), min_candidate_iou=0.0)
    lookup_df = expand_anchor_rows_for_lookup(anchor_df, int(args.trajectory_anchor_search_radius))

    before_cols = select_local_overlap_anchor_matches(
        "SORT", before_sctc, gt_sctc, lookup_df, cfg, "before"
    )
    after_cols = select_local_overlap_anchor_matches(
        "LiveCellX", after_sctc, gt_sctc, lookup_df, cfg, "after"
    )
    mapping = anchor_df.merge(before_cols, on="gt_tid", how="left")
    mapping = mapping.merge(after_cols, on="gt_tid", how="left")
    if ultrack_sctc is not None:
        ultrack_cols = select_local_overlap_anchor_matches(
            "Ultrack", ultrack_sctc, gt_sctc, lookup_df, cfg, "ultrack"
        )
        mapping = mapping.merge(ultrack_cols, on="gt_tid", how="left")
    mapping["before_anchor_matched"] = mapping["before_anchor_found"].fillna(False)
    mapping["after_anchor_matched"] = mapping["after_anchor_found"].fillna(False)
    if "ultrack_anchor_found" in mapping.columns:
        mapping["ultrack_anchor_matched"] = mapping["ultrack_anchor_found"].fillna(False)
    return mapping


TRAJECTORY_FEATURE_SPECS = [
    ("area", "Area", "log(1 + area)", np.log1p),
    ("perimeter", "Perimeter", "Pixels", None),
    ("eccentricity", "Eccentricity", "0 circular, 1 elongated", None),
    ("solidity", "Solidity", "Area / convex hull", None),
    ("aspect_ratio", "Aspect ratio", "Width / height", None),
]


def trajectory_folder_name(gt_tid: int, role: str) -> str:
    return f"gt_traj_{int(gt_tid):04d}_{str(role)}"


def plot_single_trajectory_feature(
    series_df: pd.DataFrame,
    mapping_row: pd.Series,
    feature: str,
    title: str,
    ylabel: str,
    transform,
    traj_dir: Path,
    methods: Sequence[str] = TRAJECTORY_METHOD_ORDER,
    filename_suffix: str = "",
    figsize=(5.6, 3.0),
    vertical_line_time=None,
    xticks=None,
    line_style_overrides=None,
    method_color_overrides=None,
) -> None:
    gt_tid = int(mapping_row["gt_tid"])
    role = str(mapping_row["gt_role"])
    anchor_time = int(mapping_row["time"])
    fig, ax = plt.subplots(figsize=figsize)
    handles = []
    labels = []

    for method in methods:
        sub = series_df[series_df["method"] == method].sort_values("time")
        if sub.empty or feature not in sub.columns:
            continue
        y = sub[feature].astype(float)
        if transform is not None:
            y = transform(y)
        style = dict(TRAJECTORY_LINE_STYLES[method])
        if line_style_overrides is not None:
            style.update(line_style_overrides.get(method, {}))
        method_color = TRAJECTORY_METHOD_COLORS[method]
        if method_color_overrides is not None:
            method_color = method_color_overrides.get(method, method_color)
        line, = ax.plot(
            sub["time"],
            y,
            color=method_color,
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=style["markersize"],
            alpha=style.get("alpha", 0.55),
            zorder=style["zorder"],
            label=method,
        )
        handles.append(line)
        labels.append(method)

    line_time = anchor_time if vertical_line_time is None else int(vertical_line_time)
    ax.axvline(line_time, color="#6f6f6f", linewidth=0.7, linestyle="--", alpha=0.55)
    ax.set_title(f"{title}: GT {gt_tid} ({role})", fontweight="bold", loc="left")
    ax.set_xlabel("Frame")
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xlim(min(xticks), max(xticks))
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        labelspacing=0.35,
        handlelength=2.4,
    )
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(traj_dir / f"{feature}{filename_suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def contour_vector_from_sc(sc, contour_num_points: int) -> Optional[np.ndarray]:
    """Return a centered, scale-normalized contour vector for shape PCA.

    This is the active-shape-model feature step: a cell mask contour is
    converted to a fixed-length, axis-aligned landmark sequence, then flattened
    for PCA.  Translation and global scale are removed so PC1/PC2 describe
    shape changes rather than image position or absolute cell size.
    """
    if sc is None or getattr(sc, "contour", None) is None:
        return None
    try:
        points = np.asarray(sc.contour, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            return None
        contour = Contour(points=points, units="pixels")
        contour.resample(num_points=int(contour_num_points))
        contour.axis_align()
        aligned = np.asarray(contour.points, dtype=float)
        if aligned.ndim != 2 or aligned.shape != (int(contour_num_points), 2):
            return None
        aligned = aligned - np.nanmean(aligned, axis=0, keepdims=True)
        scale = float(np.sqrt(np.nanmean(np.sum(aligned**2, axis=1))))
        if not np.isfinite(scale) or scale <= 0:
            return None
        aligned = aligned / scale
        vec = aligned.reshape(-1)
        if not np.isfinite(vec).all():
            return None
        return vec
    except Exception:
        return None


def shape_pca_input_rows_for_gt_span(
    method: str,
    sctc,
    track_id: Optional[int],
    start_t: int,
    end_t: int,
    contour_num_points: int,
) -> List[dict]:
    rows: List[dict] = []
    if track_id is None or pd.isna(track_id) or int(track_id) not in sctc.track_id_to_trajectory:
        return rows
    track_id = int(track_id)
    traj = sctc.get_trajectory(track_id)
    times = set(int(t) for t in traj.times)
    for t in range(int(start_t), int(end_t) + 1):
        if t not in times:
            continue
        sc = traj.get_sc(t)
        vec = contour_vector_from_sc(sc, contour_num_points=contour_num_points)
        if vec is None:
            continue
        rows.append({"method": method, "track_id": track_id, "time": int(t), "_shape_vector": vec})
    return rows


def build_shape_pca_for_mapping_row(
    mapping_row: pd.Series,
    gt_sctc,
    before_sctc,
    after_sctc,
    ultrack_sctc,
    contour_num_points: int,
    method_start: int,
    method_end: int,
) -> pd.DataFrame:
    gt_tid = int(mapping_row["gt_tid"])
    gt_start = int(mapping_row["gt_start"])
    gt_end = int(mapping_row["gt_end"])
    before_tid = mapping_row.get("before_track_id")
    after_tid = mapping_row.get("after_track_id")
    ultrack_tid = mapping_row.get("ultrack_track_id")

    rows: List[dict] = []
    rows.extend(shape_pca_input_rows_for_gt_span("GT", gt_sctc, gt_tid, gt_start, gt_end, contour_num_points))
    rows.extend(
        shape_pca_input_rows_for_gt_span(
            "SORT", before_sctc, before_tid, method_start, method_end, contour_num_points
        )
    )
    rows.extend(
        shape_pca_input_rows_for_gt_span(
            "LiveCellX", after_sctc, after_tid, method_start, method_end, contour_num_points
        )
    )
    if ultrack_sctc is not None:
        rows.extend(
            shape_pca_input_rows_for_gt_span(
                "Ultrack", ultrack_sctc, ultrack_tid, method_start, method_end, contour_num_points
            )
        )
    if len(rows) < 2:
        return pd.DataFrame()

    X = np.vstack([row["_shape_vector"] for row in rows])
    if X.shape[0] < 2 or X.shape[1] < 4:
        return pd.DataFrame()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    out_rows: List[dict] = []
    for row, coord in zip(rows, coords):
        out_rows.append(
            {
                "gt_tid": gt_tid,
                "gt_role": mapping_row["gt_role"],
                "method": row["method"],
                "track_id": row["track_id"],
                "time": row["time"],
                "shape_pc1": float(coord[0]),
                "shape_pc2": float(coord[1]),
                "shape_pca_explained_var_pc1": float(pca.explained_variance_ratio_[0]),
                "shape_pca_explained_var_pc2": float(pca.explained_variance_ratio_[1]),
                "contour_num_points": int(contour_num_points),
            }
        )
    return pd.DataFrame(out_rows)


def plot_single_trajectory_shape_pca(
    shape_df: pd.DataFrame,
    mapping_row: pd.Series,
    traj_dir: Path,
    methods: Sequence[str] = TRAJECTORY_METHOD_ORDER,
    filename_suffix: str = "",
    colorbar_frame_min: float = 0.0,
    colorbar_frame_max: float = 99.0,
) -> None:
    gt_tid = int(mapping_row["gt_tid"])
    role = str(mapping_row["gt_role"])
    anchor_time = int(mapping_row["time"])
    shape_df = shape_df[shape_df["method"].isin(methods)].copy()
    if shape_df.empty:
        (traj_dir / f"shape_pca{filename_suffix}_skipped.txt").write_text(
            "Shape PCA was skipped because fewer than two valid contours were available.\n"
        )
        return

    shape_df.to_csv(traj_dir / f"shape_pca_series{filename_suffix}.csv", index=False)
    frame_min = float(colorbar_frame_min)
    frame_max = float(colorbar_frame_max)
    if frame_min == frame_max:
        frame_max = frame_min + 1.0
    norm = Normalize(vmin=frame_min, vmax=frame_max)
    cmap = cm.get_cmap("viridis")

    x = shape_df["shape_pc1"].to_numpy(dtype=float)
    y = shape_df["shape_pc2"].to_numpy(dtype=float)
    x_pad = max(0.15, 0.08 * float(np.nanmax(x) - np.nanmin(x))) if len(x) else 0.15
    y_pad = max(0.15, 0.08 * float(np.nanmax(y) - np.nanmin(y))) if len(y) else 0.15
    xlim = (float(np.nanmin(x) - x_pad), float(np.nanmax(x) + x_pad))
    ylim = (float(np.nanmin(y) - y_pad), float(np.nanmax(y) + y_pad))

    fig_width = 8.8 if len(methods) == 3 else 11.5
    fig, axes = plt.subplots(1, len(methods), figsize=(fig_width, 2.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods):
        sub = shape_df[shape_df["method"] == method].sort_values("time")
        if sub.empty:
            ax.text(0.5, 0.5, "No matched cells", ha="center", va="center", transform=ax.transAxes, color="#777777")
        else:
            ax.scatter(
                sub["shape_pc1"],
                sub["shape_pc2"],
                c=sub["time"],
                cmap=cmap,
                norm=norm,
                s=18,
                marker="o",
                linewidths=0,
                edgecolors="none",
                alpha=0.82,
            )
        ax.axvline(0, color="#d8dde3", linewidth=0.5, zorder=0)
        ax.axhline(0, color="#d8dde3", linewidth=0.5, zorder=0)
        ax.set_title(method, fontweight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.set_xlabel("Shape PC1")
    axes[0].set_ylabel("Shape PC2")

    ev1 = float(shape_df["shape_pca_explained_var_pc1"].iloc[0]) * 100.0
    ev2 = float(shape_df["shape_pca_explained_var_pc2"].iloc[0]) * 100.0
    fig.suptitle(
        f"Shape PCA: GT {gt_tid} ({role}), anchor frame {anchor_time}; PC1 {ev1:.1f}%, PC2 {ev2:.1f}%",
        fontweight="bold",
        y=1.02,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.025)
    cbar.set_label("Frame")
    fig.savefig(traj_dir / f"shape_pca{filename_suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_single_trajectory_readme(series_df: pd.DataFrame, mapping_row: pd.Series, traj_dir: Path) -> None:
    gt_tid = int(mapping_row["gt_tid"])
    role = str(mapping_row["gt_role"])
    anchor_time = int(mapping_row["time"])
    before_tid = mapping_row.get("before_track_id")
    after_tid = mapping_row.get("after_track_id")
    ultrack_tid = mapping_row.get("ultrack_track_id")
    before_time = mapping_row.get("before_anchor_time_used", np.nan)
    after_time = mapping_row.get("after_anchor_time_used", np.nan)
    ultrack_time = mapping_row.get("ultrack_anchor_time_used", np.nan)
    before_text = "NA" if before_tid is None or pd.isna(before_tid) else str(int(before_tid))
    after_text = "NA" if after_tid is None or pd.isna(after_tid) else str(int(after_tid))
    ultrack_text = "NA" if ultrack_tid is None or pd.isna(ultrack_tid) else str(int(ultrack_tid))
    before_time_text = "NA" if pd.isna(before_time) else str(int(before_time))
    after_time_text = "NA" if pd.isna(after_time) else str(int(after_time))
    ultrack_time_text = "NA" if pd.isna(ultrack_time) else str(int(ultrack_time))
    lines = [
        f"# GT trajectory {gt_tid} ({role})",
        "",
        f"- Anchor frame: `{anchor_time}`",
        f"- SORT track ID: `{before_text}` at frame `{before_time_text}`",
        f"- LiveCellX track ID: `{after_text}` at frame `{after_time_text}`",
        f"- Ultrack track ID: `{ultrack_text}` at frame `{ultrack_time_text}`",
        "",
        "Figures in this folder:",
        "",
    ]
    for feature, title, _, _ in TRAJECTORY_FEATURE_SPECS:
        lines.append(f"- `{feature}.png`: {title}, GT/SORT/LiveCellX only.")
        lines.append(f"- `{feature}_with_ultrack.png`: {title}, GT/SORT/LiveCellX/Ultrack.")
    lines.append(
        "- `shape_pca.png`: active-shape-model PCA scatter for GT/SORT/LiveCellX. Cell contours are resampled, axis-aligned, centered, scale-normalized, projected onto the first two local shape PCs, and colored by frame. All trajectory cases use the same viridis frame-color scale for the configured analysis window."
    )
    lines.append(
        "- `shape_pca_with_ultrack.png`: the same active-shape PCA scatter with Ultrack added as the fourth panel."
    )
    lines.extend(
        [
            "",
            "`feature_series.csv` contains the source values for all feature line plots. `shape_pca_series.csv` and `shape_pca_series_with_ultrack.csv` contain the source PC1/PC2 coordinates when enough valid contours were available.",
        ]
    )
    (traj_dir / "README.md").write_text("\n".join(lines))
    series_df.to_csv(traj_dir / "feature_series.csv", index=False)


def plot_single_gt_anchored_trajectory(series_df: pd.DataFrame, mapping_row: pd.Series, out_dir: Path) -> None:
    gt_tid = int(mapping_row["gt_tid"])
    role = str(mapping_row["gt_role"])
    traj_dir = out_dir / trajectory_folder_name(gt_tid, role)
    traj_dir.mkdir(parents=True, exist_ok=True)
    series_df.to_csv(traj_dir / "feature_series.csv", index=False)
    for feature, title, ylabel, transform in TRAJECTORY_FEATURE_SPECS:
        plot_single_trajectory_feature(
            series_df,
            mapping_row,
            feature,
            title,
            ylabel,
            transform,
            traj_dir,
            methods=TRAJECTORY_METHOD_ORDER,
            filename_suffix="",
        )
        plot_single_trajectory_feature(
            series_df,
            mapping_row,
            feature,
            title,
            ylabel,
            transform,
            traj_dir,
            methods=TRAJECTORY_METHOD_ORDER_WITH_ULTRACK,
            filename_suffix="_with_ultrack",
        )
    write_single_trajectory_readme(series_df, mapping_row, traj_dir)



def build_no_gt_shape_pca(
    before_sctc,
    after_sctc,
    min_frame: int,
    max_frame: int,
    contour_num_points: int,
) -> pd.DataFrame:
    """Fit the same local active-shape PCA used by GT-anchored cases, without GT."""
    rows: List[dict] = []
    rows.extend(
        shape_pca_input_rows_for_gt_span(
            "SORT", before_sctc, NO_GT_SORT_TRACK_ID, min_frame, max_frame, contour_num_points
        )
    )
    rows.extend(
        shape_pca_input_rows_for_gt_span(
            "LiveCellX", after_sctc, NO_GT_LIVECELLX_TRACK_ID, min_frame, max_frame, contour_num_points
        )
    )
    if len(rows) < 2:
        return pd.DataFrame()

    vectors = np.vstack([row["_shape_vector"] for row in rows])
    if vectors.shape[0] < 2 or vectors.shape[1] < 4:
        return pd.DataFrame()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    return pd.DataFrame(
        [
            {
                "method": row["method"],
                "track_id": int(row["track_id"]),
                "time": int(row["time"]),
                "shape_pc1": float(coord[0]),
                "shape_pc2": float(coord[1]),
                "shape_pca_explained_var_pc1": float(pca.explained_variance_ratio_[0]),
                "shape_pca_explained_var_pc2": float(pca.explained_variance_ratio_[1]),
                "contour_num_points": int(contour_num_points),
            }
            for row, coord in zip(rows, coords)
        ]
    )


def plot_no_gt_shape_pca(shape_df: pd.DataFrame, case_dir: Path, min_frame: int, max_frame: int) -> None:
    if shape_df.empty:
        (case_dir / "shape_pca_skipped.txt").write_text(
            "Shape PCA was skipped because fewer than two valid contours were available.\n"
        )
        return
    shape_df.to_csv(case_dir / "shape_pca_series.csv", index=False)
    norm = Normalize(vmin=float(min_frame), vmax=float(max_frame))
    cmap = cm.get_cmap("viridis")
    methods = ["SORT", "LiveCellX"]

    x = shape_df["shape_pc1"].to_numpy(dtype=float)
    y = shape_df["shape_pc2"].to_numpy(dtype=float)
    x_pad = max(0.15, 0.08 * float(np.nanmax(x) - np.nanmin(x)))
    y_pad = max(0.15, 0.08 * float(np.nanmax(y) - np.nanmin(y)))
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.8), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        sub = shape_df[shape_df["method"] == method].sort_values("time")
        if sub.empty:
            ax.text(0.5, 0.5, "No cells", ha="center", va="center", transform=ax.transAxes, color="#777777")
        else:
            ax.scatter(
                sub["shape_pc1"],
                sub["shape_pc2"],
                c=sub["time"],
                cmap=cmap,
                norm=norm,
                s=18,
                marker="o",
                linewidths=0,
                edgecolors="none",
                alpha=0.82,
            )
        ax.axvline(0, color="#d8dde3", linewidth=0.5, zorder=0)
        ax.axhline(0, color="#d8dde3", linewidth=0.5, zorder=0)
        ax.set_title(method, fontweight="bold")
        ax.set_xlim(float(np.nanmin(x) - x_pad), float(np.nanmax(x) + x_pad))
        ax.set_ylim(float(np.nanmin(y) - y_pad), float(np.nanmax(y) + y_pad))
        ax.set_xlabel("Shape PC1")
        ax.grid(False)
    axes[0].set_ylabel("Shape PC2")
    ev1 = float(shape_df["shape_pca_explained_var_pc1"].iloc[0]) * 100.0
    ev2 = float(shape_df["shape_pca_explained_var_pc2"].iloc[0]) * 100.0
    fig.suptitle(
        f"Shape PCA without GT; PC1 {ev1:.1f}%, PC2 {ev2:.1f}%",
        fontweight="bold",
        y=1.02,
    )
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("Frame")
    fig.savefig(case_dir / "shape_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_no_gt_trajectory_case(before_sctc, after_sctc, out_dir: Path, args: argparse.Namespace) -> None:
    """Generate the fixed LiveCellX 440 versus SORT 413 case with no GT curve."""
    if NO_GT_SORT_TRACK_ID not in before_sctc.track_id_to_trajectory:
        raise KeyError(f"SORT trajectory {NO_GT_SORT_TRACK_ID} is absent from the full SORT collection")
    if NO_GT_LIVECELLX_TRACK_ID not in after_sctc.track_id_to_trajectory:
        raise KeyError(f"LiveCellX trajectory {NO_GT_LIVECELLX_TRACK_ID} is absent from the full LiveCellX collection")

    min_frame = max(0, int(args.min_frame))
    max_frame = min(99, int(args.max_frame) if args.max_frame is not None else 99)
    case_dir = out_dir / NO_GT_CASE_FOLDER
    case_dir.mkdir(parents=True, exist_ok=True)
    rows = extract_sctc_series_for_gt_span(
        "SORT", before_sctc, NO_GT_SORT_TRACK_ID, min_frame, max_frame, args.exact_mask_area
    )
    rows.extend(
        extract_sctc_series_for_gt_span(
            "LiveCellX", after_sctc, NO_GT_LIVECELLX_TRACK_ID, min_frame, max_frame, args.exact_mask_area
        )
    )
    series_df = pd.DataFrame(rows)
    series_df.to_csv(case_dir / "feature_series.csv", index=False)

    for feature, title, ylabel, transform in TRAJECTORY_FEATURE_SPECS:
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        for method in ("SORT", "LiveCellX"):
            sub = series_df[series_df["method"] == method].sort_values("time")
            values = sub[feature].astype(float)
            if transform is not None:
                values = transform(values)
            style = TRAJECTORY_LINE_STYLES[method]
            ax.plot(
                sub["time"],
                values,
                color=TRAJECTORY_METHOD_COLORS[method],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=style["markersize"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                label=method,
            )
        ax.set_title(f"{title}: LiveCellX 440 vs SORT 413 (no GT)", fontweight="bold", loc="left")
        ax.set_xlabel("Frame")
        ax.set_ylabel(ylabel)
        ax.grid(False)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        fig.tight_layout(rect=(0, 0, 0.78, 1))
        fig.savefig(case_dir / f"{feature}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    shape_df = build_no_gt_shape_pca(
        before_sctc,
        after_sctc,
        min_frame,
        max_frame,
        contour_num_points=int(args.shape_pca_contour_points),
    )
    plot_no_gt_shape_pca(shape_df, case_dir, min_frame, max_frame)
    (case_dir / "README.md").write_text(
        "# Trajectory case without GT\n\n"
        "- LiveCellX trajectory: `440`\n"
        "- SORT trajectory: `413`\n"
        f"- Frames: `{min_frame}-{max_frame}`\n"
        "- These IDs were specified directly; no GT anchor or rematching was used.\n"
        "- The plots use the same feature extraction, styles, and active-shape PCA logic as the GT-anchored cases, with the GT series omitted.\n"
    )


def write_trajectory_results_readme(out_dir: Path, mapping_df: pd.DataFrame) -> None:
    lines = [
        "# GT-anchored trajectory downstream results",
        "",
        "These figures are trajectory-based, not frame-aggregate-based.",
        "",
        "For each annotated GT trajectory, one anchor cell is used to identify the corresponding SORT and LiveCellX track IDs:",
        "",
        "- Mother trajectory: the first GT cell is used as the anchor.",
        "- Daughter trajectory: the first GT cell is used as the anchor.",
        "- The anchor cell region is used to select the local best-overlap SORT and LiveCellX trajectory IDs. Selection is based on the actual GT mask region, not on keeping only matches above an IoU threshold.",
        "- If no local trajectory exists at the exact anchor frame, the lookup searches forward within the same annotated GT trajectory span. By default this continues across the full annotated GT trajectory span.",
        "- The anchor IoU threshold is recorded as a diagnostic pass/fail column, but it does not remove the local best-overlap trajectory from the plot.",
        "- The GT trajectory is plotted on its annotated identity span; each matched method track ID is plotted across its full available trajectory within the evaluation window.",
        "",
        "Each GT trajectory has its own folder. Inside that folder, area, perimeter, eccentricity, solidity, and aspect ratio are saved as five separate PNG figures. Each figure contains three lightly transparent lines: GT, SORT, and LiveCellX. The dashed vertical line marks the anchor frame.",
        "The additional `no_gt_livecellx_0440_sort_0413` folder uses the same feature and shape-PCA plotting logic for the manually specified LiveCellX 440 and SORT 413 trajectories over frames 0-99, but omits GT entirely.",
        "",
        "",
        "For every per-trajectory PNG, a second `_with_ultrack.png` version is saved beside it. The base file keeps GT/SORT/LiveCellX only; the `_with_ultrack` file adds the matched Ultrack trajectory.",
        "",
        "Each folder also contains `shape_pca.png`. This panel fits a local active-shape PCA model to the valid GT, SORT, and LiveCellX contours from that trajectory span. The three panels share the same PC axes, and every trajectory case uses the same viridis frame-color scale for the configured analysis window.",
        "",
        "`shape_pca_with_ultrack.png` adds Ultrack as a fourth panel while keeping the same local PC axes and shared frame colorbar.",
        "",
        "Output tables:",
        "",
        "- `gt_anchored_trajectory_mapping.csv`: GT trajectory ID, anchor frame, before/after matched track IDs, and anchor IoU values.",
        "- `before_anchor_time_used` and `after_anchor_time_used`: the actual frame used to identify the plotted before/after trajectory ID. These equal the anchor frame unless fallback search was needed.",
        "- `gt_anchored_trajectory_feature_series.csv`: source data for every per-trajectory line plot.",
        "",
        "Anchor-match summary:",
        "",
    ]
    if len(mapping_df):
        summary = pd.DataFrame(
            [
                {
                    "n_gt_trajectories": int(len(mapping_df)),
                    "before_anchor_found": int(mapping_df["before_anchor_found"].sum()),
                    "after_anchor_found": int(mapping_df["after_anchor_found"].sum()),
                    "before_anchor_iou_pass": int(mapping_df["before_anchor_pass_iou_threshold"].fillna(False).sum()),
                    "after_anchor_iou_pass": int(mapping_df["after_anchor_pass_iou_threshold"].fillna(False).sum()),
                    "ultrack_anchor_found": int(mapping_df["ultrack_anchor_found"].sum()) if "ultrack_anchor_found" in mapping_df.columns else 0,
                    "ultrack_anchor_iou_pass": int(mapping_df["ultrack_anchor_pass_iou_threshold"].fillna(False).sum()) if "ultrack_anchor_pass_iou_threshold" in mapping_df.columns else 0,
                    "mother_anchors": int((mapping_df["gt_role"] == "mother").sum()),
                    "daughter_anchors": int((mapping_df["gt_role"] == "daughter").sum()),
                }
            ]
        )
        lines.append("```")
        lines.append(summary.to_string(index=False))
        lines.append("```")
    (out_dir / "README_gt_anchored_trajectory_results.md").write_text("\n".join(lines))



def gt_trajectory_xyz(traj, min_frame: int, max_frame: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    ts: List[float] = []
    for time in sorted(int(t) for t in traj.times):
        if time < int(min_frame) or (max_frame is not None and time > int(max_frame)):
            continue
        try:
            center_y, center_x = np.asarray(traj.get_sc(time).get_center(), dtype=float)[:2]
        except Exception:
            sc = traj.get_sc(time)
            bbox = np.asarray(sc.bbox, dtype=float)
            center_y = float((bbox[0] + bbox[2]) / 2.0)
            center_x = float((bbox[1] + bbox[3]) / 2.0)
        xs.append(float(center_x))
        ys.append(float(center_y))
        ts.append(float(time))
    return np.asarray(xs), np.asarray(ys), np.asarray(ts)


def style_gt_lineage_3d_axis(ax: plt.Axes, emphasized: bool = False) -> None:
    pane_color = (0.94, 0.94, 0.94, 0.58)
    axis_color = "#333333"
    label_size = 14 if emphasized else 8
    tick_size = 12 if emphasized else 7
    ax.set_xlabel("X (pixel)", labelpad=8 if emphasized else 6, fontsize=label_size, color=axis_color)
    ax.set_ylabel("Y (pixel)", labelpad=8 if emphasized else 6, fontsize=label_size, color=axis_color)
    ax.set_zlabel("Time", labelpad=7 if emphasized else 5, fontsize=label_size, color=axis_color)
    ax.tick_params(axis="x", labelsize=tick_size, pad=2, colors=axis_color)
    ax.tick_params(axis="y", labelsize=tick_size, pad=2, colors=axis_color)
    ax.tick_params(axis="z", labelsize=tick_size, pad=2, colors=axis_color)
    ax.set_facecolor("white")
    ax.view_init(elev=10, azim=-18, roll=0)
    try:
        ax.set_proj_type("persp")
    except Exception:
        pass
    try:
        ax.set_box_aspect((1.15, 1.0, 1.30))
    except Exception:
        pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane_color)
        axis.pane.set_alpha(None)
        axis.pane.set_edgecolor((0.62, 0.62, 0.62, 0.52))
        axis.line.set_color(axis_color)
        axis.line.set_linewidth(1.25 if emphasized else 0.8)
        axis._axinfo["grid"]["color"] = (0.0, 0.0, 0.0, 0.0)
        axis._axinfo["grid"]["linewidth"] = 0.0
    ax.grid(False)


def plot_gt_lineages_4f_original_by_mother(
    gt_sctc,
    downstream_dir: Path,
    min_frame: int = 0,
    max_frame: Optional[int] = 99,
    before_sctc=None,
    after_sctc=None,
) -> None:
    """Plot all GT lineages and two GT-matched lineage examples for each method."""
    out_dir = downstream_dir / "gt_lineage_4f_original"
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        (out_dir / f"fig_gt_lineages_4f_original{suffix}").unlink(missing_ok=True)
    (out_dir / "fig_gt_lineages_4f_original_source_data.csv").unlink(missing_ok=True)
    for old_selected_path in out_dir.glob("fig_gt_lineages_4f_original_mothers_*"):
        old_selected_path.unlink()

    gt_ids = relation_ids_from_gt(gt_sctc)
    edges_df = lineage_edges_from_gt(
        gt_sctc,
        gt_ids,
        min_frame=int(min_frame),
        max_frame=max_frame,
    )
    if edges_df.empty:
        raise ValueError("No GT mother-daughter edges were available for the Fig. 4f-style lineage plot.")

    selected_mother_ids = {1528, 1534}
    selected_edges_df = edges_df[edges_df["mother_gt_tid"].astype(int).isin(selected_mother_ids)].copy()
    found_mother_ids = set(selected_edges_df["mother_gt_tid"].astype(int).unique())
    missing_mother_ids = selected_mother_ids - found_mother_ids
    if missing_mother_ids:
        raise ValueError(
            f"Selected GT mothers without mother-daughter edges: {sorted(missing_mother_ids)}"
        )

    def daughter_ids_for_trajectory(source_sctc, trajectory) -> List[int]:
        daughter_ids = {int(d.track_id) for d in trajectory.daughter_trajectories}
        daughter_ids.update(
            int(x)
            for x in (trajectory.meta or {}).get("daughter_trajectory_ids", [])
            if x is not None
        )
        existing_ids = set(int(x) for x in source_sctc.track_id_to_trajectory)
        daughter_ids.intersection_update(existing_ids)
        return sorted(
            daughter_ids,
            key=lambda daughter_id: (
                min(int(t) for t in source_sctc.get_trajectory(daughter_id).times),
                daughter_id,
            ),
        )

    def render_lineages(
        source_sctc,
        mother_specs: Sequence[dict],
        output_stem: str,
        title: str,
        source_name: str,
        annotate_mother_ids: bool = False,
        emphasized: bool = False,
    ) -> None:
        mother_color = "#009E73"
        daughter_colors = ("#56B4E9", "#E45756")
        event_rows: List[dict] = []
        mother_linewidth = 5.2 if emphasized else 1.7
        daughter_linewidth = 4.5 if emphasized else 1.25
        line_alpha = 0.68 if emphasized else 0.48
        division_marker_size = 120 if emphasized else 13
        division_edge_width = 2.0 if emphasized else 0.75

        fig = plt.figure(
            figsize=(10.0, 8.2) if emphasized else (8.2, 7.2),
            constrained_layout=True,
            facecolor="white",
        )
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        role_labels_drawn = {"mother": False, "daughter1": False, "daughter2": False}

        for spec in mother_specs:
            mother_id = int(spec["mother_tid"])
            mother = source_sctc.get_trajectory(mother_id)
            daughter_ids = daughter_ids_for_trajectory(source_sctc, mother)
            if len(daughter_ids) > 2:
                print(f"[warn] {source_name} mother {mother_id} has {len(daughter_ids)} daughters; plotting the first two")
            daughter_ids = daughter_ids[:2]
            mother_x, mother_y, mother_t = gt_trajectory_xyz(mother, int(min_frame), max_frame)
            if len(mother_t) == 0:
                print(f"[warn] {source_name} mother {mother_id} is outside the plotting window")
                continue

            daughter_series = []
            for daughter_id in daughter_ids:
                daughter = source_sctc.get_trajectory(int(daughter_id))
                daughter_x, daughter_y, daughter_t = gt_trajectory_xyz(
                    daughter,
                    int(min_frame),
                    max_frame,
                )
                if len(daughter_t) == 0:
                    continue
                daughter_series.append((int(daughter_id), daughter_x, daughter_y, daughter_t))

            if len(daughter_series) < len(daughter_ids):
                print(
                    f"[warn] {source_name} mother {mother_id} has a daughter outside "
                    "the plotting window"
                )
            if not daughter_ids:
                print(f"[info] {source_name} matched mother {mother_id} has no stored daughters; plotting mother only")

            division_x = float(mother_x[-1])
            division_y = float(mother_y[-1])
            division_t = float(mother_t[-1])
            ax.plot(
                mother_x,
                mother_y,
                mother_t,
                color=mother_color,
                linewidth=mother_linewidth,
                alpha=line_alpha,
                label="Mother" if not role_labels_drawn["mother"] else None,
                zorder=2,
            )
            role_labels_drawn["mother"] = True
            if daughter_series:
                ax.scatter(
                    [division_x],
                    [division_y],
                    [division_t],
                    s=division_marker_size,
                    facecolors="white",
                    edgecolors="#222222",
                    linewidths=division_edge_width,
                    depthshade=False,
                    zorder=6,
                )
            if annotate_mother_ids:
                ax.text(
                    division_x,
                    division_y,
                    division_t,
                    f" {mother_id}",
                    color=mother_color,
                    fontsize=6,
                    fontweight="bold",
                    zorder=7,
                )
            for daughter_index, (daughter_id, daughter_x, daughter_y, daughter_t) in enumerate(daughter_series):
                branch_x = np.concatenate(([division_x], daughter_x))
                branch_y = np.concatenate(([division_y], daughter_y))
                branch_t = np.concatenate(([division_t], daughter_t))
                role_key = f"daughter{daughter_index + 1}"
                ax.plot(
                    branch_x,
                    branch_y,
                    branch_t,
                    color=daughter_colors[daughter_index],
                    linewidth=daughter_linewidth,
                    alpha=line_alpha,
                    label=f"Daughter {daughter_index + 1}" if not role_labels_drawn[role_key] else None,
                    zorder=3 + daughter_index,
                )
                role_labels_drawn[role_key] = True

            event_rows.append(
                {
                    "source": source_name,
                    "gt_mother_tid": int(spec["gt_mother_tid"]),
                    "mother_tid": mother_id,
                    "match_anchor_time": spec.get("anchor_time", np.nan),
                    "match_anchor_iou": spec.get("anchor_iou", np.nan),
                    "daughter1_tid": daughter_series[0][0] if len(daughter_series) >= 1 else np.nan,
                    "daughter2_tid": daughter_series[1][0] if len(daughter_series) >= 2 else np.nan,
                    "division_time": int(division_t) if daughter_series else np.nan,
                    "mother_start": int(mother_t.min()),
                    "mother_end": int(mother_t.max()),
                    "daughter1_start": int(daughter_series[0][3].min()) if len(daughter_series) >= 1 else np.nan,
                    "daughter2_start": int(daughter_series[1][3].min()) if len(daughter_series) >= 2 else np.nan,
                }
            )

        ax.set_title(title, fontsize=18 if emphasized else 11, fontweight="bold", pad=14 if emphasized else 8)
        style_gt_lineage_3d_axis(ax, emphasized=emphasized)
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=14 if emphasized else 8,
            handlelength=3.0 if emphasized else 2.0,
        )
        save_figure(fig, out_dir / output_stem)
        pd.DataFrame(event_rows).to_csv(out_dir / f"{output_stem}_source_data.csv", index=False)
        print(f"[done] saved {output_stem} with {len(event_rows)} mothers to {out_dir}")

    def gt_specs(mother_ids: Sequence[int]) -> List[dict]:
        return [
            {
                "gt_mother_tid": int(mother_id),
                "mother_tid": int(mother_id),
                "anchor_time": np.nan,
                "anchor_iou": 1.0,
            }
            for mother_id in sorted(set(int(x) for x in mother_ids))
        ]

    def match_method_mothers(source_sctc, source_name: str) -> List[dict]:
        anchor_rows = []
        for gt_mother_id in sorted(selected_mother_ids):
            gt_mother = gt_sctc.get_trajectory(int(gt_mother_id))
            valid_times = [
                int(t)
                for t in gt_mother.times
                if int(t) >= int(min_frame) and (max_frame is None or int(t) <= int(max_frame))
            ]
            if not valid_times:
                continue
            anchor_rows.append((int(gt_mother_id), int(min(valid_times))))

        time_index = build_time_index(source_sctc, {time for _, time in anchor_rows})
        cfg = MatchConfig(iou_threshold=0.0, min_candidate_iou=0.0)
        area_cache: Dict[int, float] = {}
        matched_specs: List[dict] = []
        for gt_mother_id, anchor_time in anchor_rows:
            gt_sc = gt_sctc.get_trajectory(gt_mother_id).get_sc(anchor_time)
            target_tid, anchor_iou = best_sctc_match(
                gt_sc,
                time_index.get(anchor_time, []),
                cfg,
                area_cache,
            )
            if target_tid is None:
                print(f"[warn] no {source_name} mother matched GT mother {gt_mother_id} at frame {anchor_time}")
                continue
            matched_specs.append(
                {
                    "gt_mother_tid": gt_mother_id,
                    "mother_tid": int(target_tid),
                    "anchor_time": anchor_time,
                    "anchor_iou": float(anchor_iou),
                }
            )
            print(
                f"[lineage match] {source_name}: GT mother {gt_mother_id} -> "
                f"track {int(target_tid)} at frame {anchor_time}, IoU={anchor_iou:.3f}"
            )
        return matched_specs

    all_gt_mother_ids = sorted(set(int(x) for x in edges_df["mother_gt_tid"]))
    render_lineages(
        gt_sctc,
        gt_specs(all_gt_mother_ids),
        "fig_gt_lineages_4f_original_all",
        "GT mother-daughter lineage trajectories",
        "GT",
        annotate_mother_ids=True,
    )
    render_lineages(
        gt_sctc,
        gt_specs(sorted(selected_mother_ids)),
        "fig_gt_lineages_4f_original_selected",
        "Selected GT mother-daughter lineage trajectories",
        "GT",
        emphasized=True,
    )

    if before_sctc is not None:
        render_lineages(
            before_sctc,
            match_method_mothers(before_sctc, "SORT"),
            "fig_sort_lineages_4f_original_selected",
            "Selected SORT lineage trajectories",
            "SORT",
            emphasized=True,
        )
    if after_sctc is not None:
        render_lineages(
            after_sctc,
            match_method_mothers(after_sctc, "LiveCellX"),
            "fig_livecellx_lineages_4f_original_selected",
            "Selected LiveCellX lineage trajectories",
            "LiveCellX",
            emphasized=True,
        )

def plot_gt_anchored_trajectory_results(args: argparse.Namespace, downstream_dir: Path) -> None:
    out_dir = downstream_dir / str(args.trajectory_results_subdir)
    clean_trajectory_results_dir(out_dir)

    gt_sctc = load_sctc(resolve_path(Path(args.gt_path)))
    before_sctc = load_sctc(resolve_path(Path(args.before_path), Path(args.before_fallback_path)))
    after_sctc = load_sctc(resolve_path(Path(args.after_path)))
    plot_gt_lineages_4f_original_by_mother(
        gt_sctc,
        downstream_dir,
        min_frame=int(args.min_frame),
        max_frame=args.max_frame,
        before_sctc=before_sctc,
        after_sctc=after_sctc,
    )
    ultrack_path = resolve_path(Path(args.ultrack_zarr_path))
    ultrack_sctc = None
    if is_sctc_json_path(ultrack_path):
        ultrack_sctc = load_sctc(ultrack_path)
    else:
        print(
            f"[warn] trajectory-level Ultrack overlays require an SCTC JSON; got {ultrack_path}. "
            "Saving only GT/SORT/LiveCellX trajectory folders."
        )

    gt_ids = relation_ids_from_gt(gt_sctc)
    anchor_df = build_gt_trajectory_anchor_table(gt_sctc, gt_ids, min_frame=int(args.min_frame), max_frame=args.max_frame)
    if anchor_df.empty:
        raise ValueError("No GT trajectory anchors found for trajectory-based downstream analysis.")

    mapping_df = build_gt_anchored_trajectory_mapping(gt_sctc, before_sctc, after_sctc, ultrack_sctc, anchor_df, args)
    mapping_df.drop(columns=["_gt_times"], errors="ignore").to_csv(
        out_dir / "gt_anchored_trajectory_mapping.csv",
        index=False,
    )

    shared_colorbar_min = float(args.min_frame)
    if args.max_frame is not None:
        shared_colorbar_max = float(args.max_frame)
    else:
        all_sctcs = [gt_sctc, before_sctc, after_sctc]
        if ultrack_sctc is not None:
            all_sctcs.append(ultrack_sctc)
        shared_colorbar_max = max(
            float(max(int(t) for t in traj.times))
            for sctc in all_sctcs
            for _, traj in sctc
            if traj.times
        )

    series_rows: List[dict] = []
    shape_rows: List[dict] = []
    for _, mapping_row in mapping_df.iterrows():
        gt_tid = int(mapping_row["gt_tid"])
        gt_start = int(mapping_row["gt_start"])
        gt_end = int(mapping_row["gt_end"])
        before_tid = mapping_row.get("before_track_id")
        after_tid = mapping_row.get("after_track_id")
        ultrack_tid = mapping_row.get("ultrack_track_id")
        method_start = int(args.min_frame)
        if args.max_frame is not None:
            method_end = int(args.max_frame)
        else:
            method_end_candidates = [gt_end]
            for method_sctc, method_tid in (
                (before_sctc, before_tid),
                (after_sctc, after_tid),
                (ultrack_sctc, ultrack_tid),
            ):
                if (
                    method_sctc is not None
                    and method_tid is not None
                    and not pd.isna(method_tid)
                    and int(method_tid) in method_sctc.track_id_to_trajectory
                ):
                    method_times = method_sctc.get_trajectory(int(method_tid)).times
                    if method_times:
                        method_end_candidates.append(max(int(t) for t in method_times))
            method_end = max(method_end_candidates)
        per_traj_rows: List[dict] = []
        per_traj_rows.extend(
            extract_sctc_series_for_gt_span(
                "GT",
                gt_sctc,
                gt_tid,
                gt_start,
                gt_end,
                exact_mask_area=args.exact_mask_area,
            )
        )
        per_traj_rows.extend(
            extract_sctc_series_for_gt_span(
                "SORT",
                before_sctc,
                before_tid,
                method_start,
                method_end,
                exact_mask_area=args.exact_mask_area,
            )
        )
        per_traj_rows.extend(
            extract_sctc_series_for_gt_span(
                "LiveCellX",
                after_sctc,
                after_tid,
                method_start,
                method_end,
                exact_mask_area=args.exact_mask_area,
            )
        )
        if ultrack_sctc is not None:
            per_traj_rows.extend(
                extract_sctc_series_for_gt_span(
                    "Ultrack",
                    ultrack_sctc,
                    ultrack_tid,
                    method_start,
                    method_end,
                    exact_mask_area=args.exact_mask_area,
                )
            )
        for row in per_traj_rows:
            row.update(
                {
                    "gt_tid": gt_tid,
                    "gt_role": mapping_row["gt_role"],
                    "anchor_time": int(mapping_row["time"]),
                    "before_anchor_iou": float(mapping_row.get("before_anchor_iou", np.nan)),
                    "after_anchor_iou": float(mapping_row.get("after_anchor_iou", np.nan)),
                }
            )
        traj_series_df = pd.DataFrame(per_traj_rows)
        series_rows.extend(per_traj_rows)
        plot_single_gt_anchored_trajectory(traj_series_df, mapping_row, out_dir)
        traj_dir = out_dir / trajectory_folder_name(gt_tid, str(mapping_row["gt_role"]))
        shape_df = build_shape_pca_for_mapping_row(
            mapping_row,
            gt_sctc,
            before_sctc,
            after_sctc,
            None,
            contour_num_points=int(args.shape_pca_contour_points),
            method_start=method_start,
            method_end=method_end,
        )
        shape_df_with_ultrack = build_shape_pca_for_mapping_row(
            mapping_row,
            gt_sctc,
            before_sctc,
            after_sctc,
            ultrack_sctc,
            contour_num_points=int(args.shape_pca_contour_points),
            method_start=method_start,
            method_end=method_end,
        )
        if not shape_df.empty:
            tmp = shape_df.copy()
            tmp["figure_variant"] = "without_ultrack"
            shape_rows.extend(tmp.to_dict("records"))
        if not shape_df_with_ultrack.empty and ultrack_sctc is not None:
            tmp = shape_df_with_ultrack.copy()
            tmp["figure_variant"] = "with_ultrack"
            shape_rows.extend(tmp.to_dict("records"))
        plot_single_trajectory_shape_pca(
            shape_df,
            mapping_row,
            traj_dir,
            methods=TRAJECTORY_METHOD_ORDER,
            filename_suffix="",
            colorbar_frame_min=shared_colorbar_min,
            colorbar_frame_max=shared_colorbar_max,
        )
        if ultrack_sctc is not None:
            plot_single_trajectory_shape_pca(
                shape_df_with_ultrack,
                mapping_row,
                traj_dir,
                methods=TRAJECTORY_METHOD_ORDER_WITH_ULTRACK,
                filename_suffix="_with_ultrack",
                colorbar_frame_min=shared_colorbar_min,
                colorbar_frame_max=shared_colorbar_max,
            )

    plot_no_gt_trajectory_case(before_sctc, after_sctc, out_dir, args)

    series_df = pd.DataFrame(series_rows)
    series_df.to_csv(out_dir / "gt_anchored_trajectory_feature_series.csv", index=False)
    if shape_rows:
        pd.DataFrame(shape_rows).to_csv(out_dir / "gt_anchored_trajectory_shape_pca_series.csv", index=False)
    write_trajectory_results_readme(out_dir, mapping_df)
    print(f"[done] saved GT-anchored trajectory plots to {out_dir}")


def sample_cells_for_embedding(
    cell_df: pd.DataFrame,
    track_df: pd.DataFrame,
    max_per_method: int,
    seed: int,
) -> pd.DataFrame:
    merged = cell_df.merge(
        track_df[
            [
                "method",
                "track_id",
                "length",
                "vacancy_rate",
                "area_cv",
                "mean_area_abs_log_change",
                "mean_step_displacement",
            ]
        ],
        on=["method", "track_id"],
        how="left",
        suffixes=("", "_track"),
    )
    parts = []
    rng = np.random.default_rng(seed)
    for method in METHOD_ORDER:
        sub = merged[merged["method"] == method].copy()
        if len(sub) > max_per_method:
            idx = rng.choice(sub.index.to_numpy(), size=max_per_method, replace=False)
            sub = sub.loc[idx]
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def compute_embedding(sample_df: pd.DataFrame, mode: str, seed: int) -> Tuple[pd.DataFrame, str, float]:
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    feature_cols = [
        "area",
        "bbox_area",
        "height",
        "width",
        "aspect_ratio",
        "length",
        "vacancy_rate",
        "area_cv",
        "mean_area_abs_log_change",
        "mean_step_displacement",
    ]
    X = sample_df[feature_cols].copy()
    for col in ["area", "bbox_area", "height", "width", "length", "mean_step_displacement"]:
        X[col] = np.log1p(X[col].astype(float))
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X_scaled = StandardScaler().fit_transform(X)

    method_used = "PCA"
    if mode in ("auto", "umap"):
        try:
            import umap

            reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.2, random_state=seed)
            emb = reducer.fit_transform(X_scaled)
            method_used = "UMAP"
        except Exception as exc:
            if mode == "umap":
                raise
            print(f"[warn] UMAP unavailable or failed ({exc}); falling back to PCA")
            from sklearn.decomposition import PCA

            emb = PCA(n_components=2, random_state=seed).fit_transform(X_scaled)
    else:
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=seed).fit_transform(X_scaled)

    out = sample_df.copy()
    out["embed_1"] = emb[:, 0]
    out["embed_2"] = emb[:, 1]
    labels = out["method"].astype(str).to_numpy()
    score = np.nan
    try:
        # Silhouette is expensive for very large samples.
        if len(out) > 10000:
            rng = np.random.default_rng(seed)
            idx = rng.choice(np.arange(len(out)), size=10000, replace=False)
            score = float(silhouette_score(emb[idx], labels[idx]))
        else:
            score = float(silhouette_score(emb, labels))
    except Exception:
        pass
    return out, method_used, score


def plot_embedding(embedding_df: pd.DataFrame, method_used: str, silhouette: float, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    for method in METHOD_ORDER:
        sub = embedding_df[embedding_df["method"] == method]
        ax.scatter(
            sub["embed_1"],
            sub["embed_2"],
            s=4,
            alpha=0.35,
            linewidths=0,
            color=METHOD_COLORS[method],
            label=method,
        )
    title = f"{method_used} cell-feature embedding"
    if np.isfinite(silhouette):
        title += f" (silhouette={silhouette:.2f})"
    ax.set_title(title, fontweight="bold", loc="left")
    ax.set_xlabel(f"{method_used} 1")
    ax.set_ylabel(f"{method_used} 2")
    ax.legend(frameon=False, markerscale=2.5, loc="best")
    ax.grid(False)
    save_figure(fig, out_dir / "fig_downstream_cell_feature_embedding")


def write_readme(
    out_dir: Path,
    summary_df: pd.DataFrame,
    embedding_method: str | None = None,
    silhouette: float = np.nan,
) -> None:
    lines = [
        "# Process2 downstream analysis",
        "",
        "This folder contains downstream feature and trajectory-quality analyses using the same three process2 tracking outputs as the trajectory-statistics script.",
        "",
        "Default scope: annotated GT region only, restricted to frames 0-99 unless overridden. Each measurement is anchored to the hand-annotated process2 GT lineage trajectories; unannotated cells outside those regions are ignored.",
        "",
        "## Figures",
        "",
        "- `fig_downstream_quality_summary`: rebuttal-focused continuity and artifact metrics: very short matched tracks, tracks with gaps, tracks missing >10% of their GT-region span, high area-jump tracks, high displacement tracks, and median GT-region track length.",
        "- `fig_downstream_feature_distributions`: per-cell and per-track feature distributions measured only from matched cells in the annotated GT region.",
        f"- `{TRAJECTORY_RESULTS_SUBDIR}/gt_traj_*/`: one folder per annotated GT trajectory. Each folder contains separate area, perimeter, eccentricity, solidity, and aspect-ratio figures, plus `feature_series.csv` and a small README.",
        f"- `{TRAJECTORY_RESULTS_SUBDIR}/gt_anchored_trajectory_mapping.csv`: anchor frame, before/after matched track IDs, and anchor IoU values.",
        f"- `{TRAJECTORY_RESULTS_SUBDIR}/gt_anchored_trajectory_feature_series.csv`: source data for all trajectory-based line plots.",
        "",
    ]
    if embedding_method is not None:
        lines += [
            "- `fig_downstream_cell_feature_embedding`: optional exploratory sampled cell-feature embedding colored by method.",
            "",
            "## Optional embedding",
            "",
            f"- Embedding method used: `{embedding_method}`.",
        ]
        if np.isfinite(silhouette):
            lines.append(f"- Method-label silhouette coefficient in embedding space: `{silhouette:.4f}`.")
    else:
        lines += [
            "",
            "## Optional embedding",
            "",
            "The cell-feature embedding is not generated by default. Use `--make-embedding` only as an exploratory check; it should not be used as the primary rebuttal evidence unless it supports a clear, reproducible biological interpretation.",
        ]
    lines += [
        "",
        "## Term definitions",
        "",
        "- `annotated GT region`: the subset of process2 defined by the hand-annotated GT lineage trajectories. All default downstream measurements are restricted to these GT cells and their matched method cells; unannotated movie regions are ignored.",
        "- `GT cell`: one manually annotated cell mask at one frame in the GT trajectory collection.",
        "- `matched` or `recovered` cell: a method cell/label at the same frame that overlaps the GT cell with IoU >= the matching threshold.",
        "- `IoU`: intersection-over-union overlap between the GT cell and the method cell. The reused trajectory-statistics tables use fast bounding-box IoU by default; run the trajectory-statistics script with `--exact-match-iou` before this script if you want exact rasterized mask IoU.",
        "- `method`: the tracking result being evaluated: SORT, LiveCellX, or Ultrack.",
        "- `n_trajectories`: number of unique method track IDs covering matched GT-region cells. More track IDs can indicate fragmentation if the same annotated lineage is split across labels.",
        "- `median_length`: median number of matched GT-region frames per method track ID.",
        "- `short_track_rate_len_le_3` or `very short matched tracks`: fraction of method track IDs with length <= 3 frames. Lower means fewer tiny trajectory fragments.",
        "- `fragmented_track_rate` or `tracks with gaps`: fraction of method track IDs with at least one missing internal frame between their first and last matched GT-region frames. Lower means better continuity.",
        "- `high_vacancy_rate_gt_10pct` or `tracks missing >10% of span`: fraction of method track IDs where more than 10% of frames inside the track span are missing.",
        "- `cell area`: size of the matched cell region at one frame.",
        "- `perimeter`: length of the matched cell mask boundary.",
        "- `eccentricity`: elongation of the fitted ellipse, where 0 is close to circular and values near 1 are highly elongated.",
        "- `solidity`: area divided by convex-hull area. Lower values indicate a more concave or irregular mask.",
        "- `aspect_ratio`: bounding-box width divided by bounding-box height.",
        "- `cell shape ratio`: bounding-box width divided by bounding-box height for one matched cell. Values near 1 are more square; values far from 1 are more elongated.",
        "- `area_cv` or `trajectory area variability`: coefficient of variation of cell area along one method track ID. Higher values can indicate unstable segmentation over time.",
        "- `median_area_cv`: median area variability across method track IDs.",
        "- `mean_area_abs_log_change` or `temporal area jump`: mean absolute frame-to-frame change in log cell area. Higher values indicate stronger abrupt area changes.",
        "- `high_area_jump_rate_top10pct` or `high area-jump tracks`: fraction of method track IDs in the top 10% of temporal area-jump values across all compared methods.",
        "- `step displacement`: frame-to-frame centroid movement of one track ID, in pixels per frame.",
        "- `median_step_displacement`: median frame-to-frame centroid movement across method track IDs.",
        "- `high_displacement_rate_top10pct` or `high displacement tracks`: fraction of method track IDs in the top 10% of displacement values across all compared methods.",
        "- `trajectory straightness`: net displacement divided by path length for one track ID. Values closer to 1 mean a straighter path; lower values mean a more wandering path.",
        "- `GT-anchored trajectory feature profile`: a trajectory-level line plot where the GT trajectory is first anchored to corresponding SORT and LiveCellX track IDs, then GT is shown on its annotated identity span while each selected method trajectory is shown across its full available extent within the evaluation window.",
        "- `anchor frame`: the GT cell used to identify the corresponding method track. For trajectory-feature plots, both mother and daughter trajectories use their first annotated GT cell.",
        "- `optional embedding`: an exploratory PCA/UMAP projection of sampled cell and track features. It is not generated by default and should not be treated as the main rebuttal evidence unless the biological interpretation is clear.",
        "",
        "## Notes",
        "",
        "These analyses use automatically measured tracking-result features after restricting the evaluation to the annotated GT cells/regions. The GT annotations are used only to define the evaluation region and to match corresponding cells; unannotated cells outside those regions are ignored.",
        "",
        "High area-jump and high displacement rates are defined using the top 10% threshold across all compared methods in this GT-region analysis. High-vacancy tracks are tracks missing more than 10% of frames between their first and last matched GT-region detections.",
        "",
        "## Current downstream summary",
        "",
    ]
    if not summary_df.empty:
        lines.append("```")
        lines.append(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("```")
    (out_dir / "README_process2_downstream_analysis.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.only_gt_lineage_4f:
        gt_sctc = load_sctc(resolve_path(Path(args.gt_path)))
        before_sctc = load_sctc(resolve_path(Path(args.before_path), Path(args.before_fallback_path)))
        after_sctc = load_sctc(resolve_path(Path(args.after_path)))
        plot_gt_lineages_4f_original_by_mother(
            gt_sctc,
            out_dir,
            min_frame=int(args.min_frame),
            max_frame=args.max_frame,
            before_sctc=before_sctc,
            after_sctc=after_sctc,
        )
        return
    for old_path in out_dir.glob("fig_downstream_temporal_feature_trends.*"):
        old_path.unlink()
    for old_path in out_dir.glob("fig_downstream_temporal_*.*"):
        old_path.unlink()
    cell_df, track_df, _ = ensure_feature_tables(args)

    summary_df = downstream_summary(track_df)
    summary_df.to_csv(out_dir / "process2_downstream_summary.csv", index=False)

    plot_downstream_quality_summary(
        summary_df,
        out_dir,
        methods=DOWNSTREAM_METHOD_ORDER_NO_ULTRACK,
        filename_suffix="",
    )
    plot_downstream_quality_summary(
        summary_df,
        out_dir,
        methods=METHOD_ORDER,
        filename_suffix="_with_ultrack",
    )
    plot_feature_distributions(
        cell_df,
        track_df,
        out_dir,
        methods=DOWNSTREAM_METHOD_ORDER_NO_ULTRACK,
        filename_suffix="",
    )
    plot_feature_distributions(
        cell_df,
        track_df,
        out_dir,
        methods=METHOD_ORDER,
        filename_suffix="_with_ultrack",
    )
    if not args.skip_trajectory_results:
        plot_gt_anchored_trajectory_results(args, out_dir)

    embedding_method = None
    silhouette = np.nan
    if args.make_embedding:
        sample_df = sample_cells_for_embedding(
            cell_df,
            track_df,
            max_per_method=args.max_embedding_cells_per_method,
            seed=args.seed,
        )
        embedding_df, embedding_method, silhouette = compute_embedding(sample_df, args.embedding, args.seed)
        embedding_df.to_csv(out_dir / "process2_downstream_embedding_cells.csv", index=False)
        pd.DataFrame(
            [
                {
                    "embedding_method": embedding_method,
                    "silhouette_by_method": silhouette,
                    "n_cells": int(len(embedding_df)),
                    "max_cells_per_method": int(args.max_embedding_cells_per_method),
                }
            ]
        ).to_csv(out_dir / "process2_downstream_embedding_scores.csv", index=False)
        plot_embedding(embedding_df, embedding_method, silhouette, out_dir)
    write_readme(out_dir, summary_df, embedding_method, silhouette)
    print(f"[done] saved downstream analysis to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-path", default=str(GT_PATH))
    parser.add_argument("--before-path", default=str(DOWNSTREAM_SORT_BEFORE_FULL_PATH))
    parser.add_argument("--before-fallback-path", default=str(DOWNSTREAM_SORT_BEFORE_FULL_PATH))
    parser.add_argument("--after-path", default=str(DOWNSTREAM_SORT_AFTER_FULL_PATH))
    parser.add_argument("--ultrack-zarr-path", default=str(DOWNSTREAM_ULTRACK_FULL_PATH))
    parser.add_argument("--traj-statistics-dir", default=str(TRAJ_STATS_DIR))
    parser.add_argument("--output-dir", default=str(DOWNSTREAM_DIR))
    parser.add_argument("--min-frame", type=int, default=0, help="First frame included in downstream GT-region analysis.")
    parser.add_argument("--max-frame", type=int, default=99, help="Last frame included in downstream GT-region analysis; default keeps frames 0-99.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--trajectory-results-subdir",
        default=TRAJECTORY_RESULTS_SUBDIR,
        help="Subfolder under --output-dir for GT-anchored trajectory line plots and source CSVs.",
    )
    parser.add_argument(
        "--trajectory-anchor-min-iou",
        type=float,
        default=0.5,
        help=(
            "Diagnostic IoU threshold recorded for the GT anchor match. "
            "The trajectory plot still uses the local best-overlap trajectory ID even when this threshold is not passed."
        ),
    )
    parser.add_argument(
        "--trajectory-anchor-search-radius",
        type=int,
        default=-1,
        help=(
            "Nearby-frame fallback for trajectory ID lookup when the exact anchor frame has no local target. "
            "Trajectory-feature anchors search forward from the first annotated GT cell. "
            "-1 searches the full GT trajectory span; 0 means exact anchor only."
        ),
    )
    parser.add_argument(
        "--skip-trajectory-results",
        action="store_true",
        help="Skip the GT-anchored per-trajectory line plots.",
    )
    parser.add_argument(
        "--only-gt-lineage-4f",
        action="store_true",
        help="Generate only the original-coordinate GT, SORT, and LiveCellX lineage figures.",
    )
    parser.add_argument(
        "--shape-pca-contour-points",
        type=int,
        default=128,
        help="Number of resampled contour landmarks per cell for per-trajectory active-shape PCA plots.",
    )
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument(
        "--exact-mask-area",
        action="store_true",
        help="When recomputing SCTC features, rasterize contours for exact mask area. Default uses fast contour polygon area.",
    )
    parser.add_argument(
        "--exact-match-iou",
        action="store_true",
        help="When recomputing GT-region features, use exact rasterized mask IoU for matching. Default uses fast bbox IoU.",
    )
    parser.add_argument(
        "--make-embedding",
        action="store_true",
        help="Generate the optional exploratory cell-feature embedding. It is not part of the default rebuttal figure set.",
    )
    parser.add_argument("--embedding", choices=["auto", "pca", "umap"], default="auto")
    parser.add_argument("--max-embedding-cells-per-method", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--whole-movie",
        action="store_true",
        help="Use the old whole-movie scope instead of restricting analysis to annotated GT cells.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
