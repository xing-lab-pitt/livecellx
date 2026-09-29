#!/usr/bin/env python
"""Plot the two manually annotated Figure 4 trajectory cases.

This script reads the edited GT collection and the paired SORT and LiveCellX
collections from ``traj_collections/final_figure_4_collections``. Track IDs are
fixed by the annotation workflow; this script does not rematch trajectories or
modify any collection.

For each case it writes separate evidence figures for:

1. raw and within-trajectory standardized morphology features;
2. GT-reference active-shape PCA;
3. contour-center paths and center distance to GT;
4. raw-image key frames with GT, SORT, and LiveCellX contours.

The plotting style follows the trajectory-based outputs from
``process2_downstream_analysis.py``. All source values are also saved as CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA


SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_CODE_DIR = SCRIPT_DIR.parent
REPO_ROOT = REVISION_CODE_DIR.parent
for import_dir in (REPO_ROOT, REVISION_CODE_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from process2_ctc_style_tracking_evaluation import load_sctc  # noqa: E402
from process2_downstream_analysis import (  # noqa: E402
    TRAJECTORY_FEATURE_SPECS,
    TRAJECTORY_LINE_STYLES,
    TRAJECTORY_METHOD_COLORS,
    TRAJECTORY_METHOD_ORDER,
    extract_sctc_series_for_gt_span,
    plot_single_trajectory_feature,
    trajectory_folder_name,
)


COLLECTION_DIR = SCRIPT_DIR / "traj_collections/final_figure_4_collections"
DEFAULT_GT_PATH = COLLECTION_DIR / "livecellx_traj_gt1.json"
DEFAULT_SORT_PATH = COLLECTION_DIR / "sort_traj71_107.json"
DEFAULT_LIVECELLX_PATH = COLLECTION_DIR / "livecellx_traj102_168.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "trajectory_based_results"

FIGURE_METHOD_COLORS = {
    "GT": TRAJECTORY_METHOD_COLORS["GT"],
    "SORT": "#0072B2",
    "LiveCellX": "#E69F00",
}
FIGURE_LINE_STYLE_OVERRIDES = {
    "GT": {"linewidth": 4.5, "marker": None},
}
CONTOUR_CENTER_AXIS_LIMITS = {
    168: {"xlim": (70.0, 190.0), "ylim": (530.0, 430.0), "xticks": range(80, 181, 20)},
    102: {"xlim": (520.0, 640.0), "ylim": (370.0, 270.0), "xticks": range(530, 631, 20)},
}

CASES: List[Dict[str, int]] = [
    {
        "gt_tid": 168,
        "sort_tid": 107,
        "livecellx_tid": 168,
        "start_frame": 0,
        "end_frame": 10,
        "switch_frame": 5,
        "key_frames": [4, 5, 6],
    },
    {
        "gt_tid": 102,
        "sort_tid": 71,
        "livecellx_tid": 102,
        "start_frame": 20,
        "end_frame": 30,
        "switch_frame": 23,
        "key_frames": [22, 23, 24],
    },
]


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def require_track(sctc, track_id: int, condition: str) -> None:
    if int(track_id) not in sctc.track_id_to_trajectory:
        raise KeyError(f"{condition} trajectory {track_id} is absent from its collection")


def mapping_row(case: Dict[str, int]) -> pd.Series:
    start = int(case["start_frame"])
    end = int(case["end_frame"])
    return pd.Series(
        {
            "gt_tid": int(case["gt_tid"]),
            "gt_role": "annotated_traj",
            "time": start,
            "gt_start": start,
            "gt_end": end,
            "gt_length": end - start + 1,
            "before_track_id": int(case["sort_tid"]),
            "after_track_id": int(case["livecellx_tid"]),
        }
    )


def feature_rows(case, gt_sctc, sort_sctc, livecellx_sctc, exact_mask_area: bool) -> pd.DataFrame:
    start = int(case["start_frame"])
    end = int(case["end_frame"])
    rows = []
    for method, sctc, track_key in (
        ("GT", gt_sctc, "gt_tid"),
        ("SORT", sort_sctc, "sort_tid"),
        ("LiveCellX", livecellx_sctc, "livecellx_tid"),
    ):
        rows.extend(
            extract_sctc_series_for_gt_span(
                method,
                sctc,
                int(case[track_key]),
                start,
                end,
                exact_mask_area,
            )
        )
    return pd.DataFrame(rows)


def _normalize_shape(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    points = points - points.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(points**2, axis=1))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Contour has zero or invalid scale")
    return points / scale


def _resample_closed_contour(points: np.ndarray, num_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Contour must contain at least three finite 2D points")
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    points = points[keep]
    if len(points) < 3:
        raise ValueError("Contour has fewer than three unique points")

    closed = np.vstack([points, points[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(segment_lengths.sum())
    if not np.isfinite(perimeter) or perimeter <= 0:
        raise ValueError("Contour has zero or invalid perimeter")
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    sample_positions = np.linspace(0.0, perimeter, int(num_points), endpoint=False)
    sampled = np.column_stack(
        [np.interp(sample_positions, cumulative, closed[:, dim]) for dim in range(2)]
    )

    # Periodic low-pass smoothing suppresses pixel stair-steps while retaining
    # the coarse biological shape used by the point-distribution model.
    for _ in range(2):
        sampled = (
            np.roll(sampled, 1, axis=0) + 2.0 * sampled + np.roll(sampled, -1, axis=0)
        ) / 4.0

    # Enforce one traversal direction before cyclic landmark registration.
    x = sampled[:, 1]
    y = sampled[:, 0]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if signed_area < 0:
        sampled = sampled[::-1]
    return _normalize_shape(sampled)


def _align_shape_to_reference(shape: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cyclic landmark registration followed by rotation-only Procrustes."""
    best_shape = None
    best_error = np.inf
    for shift in range(len(shape)):
        shifted = np.roll(shape, shift, axis=0)
        u, _, vt = np.linalg.svd(shifted.T @ reference, full_matrices=False)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        aligned = shifted @ rotation
        error = float(np.sum((aligned - reference) ** 2))
        if error < best_error:
            best_error = error
            best_shape = aligned
    if best_shape is None:
        raise ValueError("Unable to align contour")
    return best_shape


def _generalized_procrustes(shapes: List[np.ndarray], max_iterations: int = 20) -> tuple:
    reference = _normalize_shape(shapes[0])
    aligned = shapes
    for _ in range(int(max_iterations)):
        aligned = [_align_shape_to_reference(shape, reference) for shape in shapes]
        updated_reference = _normalize_shape(np.mean(aligned, axis=0))
        updated_reference = _align_shape_to_reference(updated_reference, reference)
        change = float(np.sqrt(np.mean((updated_reference - reference) ** 2)))
        reference = updated_reference
        if change < 1e-7:
            break
    aligned = [_align_shape_to_reference(shape, reference) for shape in shapes]
    return reference, aligned


def build_gt_reference_shape_model(gt_sctc, contour_points: int) -> dict:
    raw_shapes = []
    source_rows = []
    for track_id, trajectory in gt_sctc:
        for sc in trajectory.get_all_scs():
            try:
                shape = _resample_closed_contour(sc.contour, contour_points)
            except Exception as exc:
                print(
                    f"[shape-model] skipping GT {track_id} frame {sc.timeframe}: {exc}",
                    flush=True,
                )
                continue
            raw_shapes.append(shape)
            source_rows.append((int(track_id), int(sc.timeframe)))
    if len(raw_shapes) < 3:
        raise ValueError("At least three valid GT contours are required for GT-reference PCA")

    reference, aligned_shapes = _generalized_procrustes(raw_shapes)
    matrix = np.vstack([shape.reshape(-1) for shape in aligned_shapes])
    pca = PCA(n_components=2)
    pca.fit(matrix)
    return {
        "reference": reference,
        "pca": pca,
        "contour_points": int(contour_points),
        "gt_sources": source_rows,
    }


def project_case_shapes(case, gt_sctc, sort_sctc, livecellx_sctc, model: dict) -> pd.DataFrame:
    rows = []
    start = int(case["start_frame"])
    end = int(case["end_frame"])
    for method, sctc, track_key in (
        ("GT", gt_sctc, "gt_tid"),
        ("SORT", sort_sctc, "sort_tid"),
        ("LiveCellX", livecellx_sctc, "livecellx_tid"),
    ):
        track_id = int(case[track_key])
        trajectory = sctc.get_trajectory(track_id)
        for frame in range(start, end + 1):
            if frame not in trajectory.timeframe_set:
                continue
            sc = trajectory.get_sc(frame)
            try:
                shape = _resample_closed_contour(sc.contour, model["contour_points"])
                aligned = _align_shape_to_reference(shape, model["reference"])
                pc1, pc2 = model["pca"].transform(aligned.reshape(1, -1))[0]
            except Exception as exc:
                print(
                    f"[shape-project] skipping {method} {track_id} frame {frame}: {exc}",
                    flush=True,
                )
                continue
            rows.append(
                {
                    "gt_tid": int(case["gt_tid"]),
                    "method": method,
                    "track_id": track_id,
                    "time": frame,
                    "shape_pc1": float(pc1),
                    "shape_pc2": float(pc2),
                }
            )
    result = pd.DataFrame(rows)
    result["shape_pca_explained_var_pc1"] = float(model["pca"].explained_variance_ratio_[0])
    result["shape_pca_explained_var_pc2"] = float(model["pca"].explained_variance_ratio_[1])
    result["contour_num_points"] = int(model["contour_points"])
    return result


def plot_gt_reference_shape_pca(
    shape_df: pd.DataFrame,
    case: Dict[str, int],
    output_path: Path,
    colorbar_min: float,
    colorbar_max: float,
    xlim: tuple,
    ylim: tuple,
) -> None:
    norm = Normalize(vmin=float(colorbar_min), vmax=float(colorbar_max))
    cmap = cm.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.8), sharex=True, sharey=True)
    for ax, method in zip(axes, TRAJECTORY_METHOD_ORDER):
        sub = shape_df[shape_df["method"] == method].sort_values("time")
        if sub.empty:
            ax.text(0.5, 0.5, "No valid contours", ha="center", va="center", transform=ax.transAxes)
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
                alpha=0.82,
            )
        ax.axvline(0, color="#d8dde3", linewidth=0.5, zorder=0)
        ax.axhline(0, color="#d8dde3", linewidth=0.5, zorder=0)
        ax.set_title(method, fontweight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Shape PC1")
        ax.grid(False)
    axes[0].set_ylabel("Shape PC2")
    ev = model_explained_variance(shape_df)
    fig.suptitle(
        f"GT-reference shape PCA: GT {case['gt_tid']}; PC1 {ev[0]:.1f}%, PC2 {ev[1]:.1f}%",
        fontweight="bold",
        y=1.02,
    )
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("Frame")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def model_explained_variance(shape_df: pd.DataFrame) -> tuple:
    return (
        float(shape_df["shape_pca_explained_var_pc1"].iloc[0]) * 100.0,
        float(shape_df["shape_pca_explained_var_pc2"].iloc[0]) * 100.0,
    )


def shape_distance_rows(shape_df: pd.DataFrame) -> pd.DataFrame:
    gt = shape_df[shape_df["method"] == "GT"][["time", "shape_pc1", "shape_pc2"]].rename(
        columns={"shape_pc1": "gt_pc1", "shape_pc2": "gt_pc2"}
    )
    rows = []
    for method in ("SORT", "LiveCellX"):
        method_df = shape_df[shape_df["method"] == method]
        merged = method_df.merge(gt, on="time", how="inner")
        for record in merged.itertuples(index=False):
            distance = np.hypot(record.shape_pc1 - record.gt_pc1, record.shape_pc2 - record.gt_pc2)
            rows.append(
                {
                    "gt_tid": int(record.gt_tid),
                    "method": method,
                    "track_id": int(record.track_id),
                    "time": int(record.time),
                    "shape_distance_to_gt": float(distance),
                }
            )
    return pd.DataFrame(rows)


def plot_shape_distance_to_gt(
    distance_df: pd.DataFrame,
    case: Dict[str, int],
    output_path: Path,
) -> None:
    start = int(case["start_frame"])
    end = int(case["end_frame"])
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for method in ("SORT", "LiveCellX"):
        sub = distance_df[distance_df["method"] == method].sort_values("time")
        if sub.empty:
            continue
        style = dict(TRAJECTORY_LINE_STYLES[method])
        ax.plot(
            sub["time"],
            sub["shape_distance_to_gt"],
            color=FIGURE_METHOD_COLORS[method],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=style["markersize"],
            alpha=style.get("alpha", 0.55),
            label=method,
        )
    ax.axvline(
        int(case["switch_frame"]),
        color="#6f6f6f",
        linewidth=0.7,
        linestyle="--",
        alpha=0.55,
    )
    ax.set_title(f"Shape distance to GT: GT {case['gt_tid']}", fontweight="bold", loc="left")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Distance in PC1-PC2")
    ax.set_xticks(list(range(start, end + 1)))
    ax.set_xlim(start, end)
    ax.grid(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def contour_center_rows(case, gt_sctc, sort_sctc, livecellx_sctc) -> pd.DataFrame:
    rows = []
    start = int(case["start_frame"])
    end = int(case["end_frame"])
    for method, sctc, track_key in (
        ("GT", gt_sctc, "gt_tid"),
        ("SORT", sort_sctc, "sort_tid"),
        ("LiveCellX", livecellx_sctc, "livecellx_tid"),
    ):
        track_id = int(case[track_key])
        trajectory = sctc.get_trajectory(track_id)
        for frame in range(start, end + 1):
            if frame not in trajectory.timeframe_set:
                continue
            sc = trajectory.get_sc(frame)
            center_row, center_col = np.asarray(sc.get_center(crop=False), dtype=float)
            image_height, image_width = np.asarray(sc.get_img()).shape[:2]
            rows.append(
                {
                    "gt_tid": int(case["gt_tid"]),
                    "method": method,
                    "track_id": track_id,
                    "time": frame,
                    "center_x": center_col,
                    "center_y": center_row,
                    "image_width": int(image_width),
                    "image_height": int(image_height),
                }
            )
    return pd.DataFrame(rows)


def complete_frame_series(sub: pd.DataFrame, start: int, end: int, column: str) -> np.ndarray:
    values = dict(zip(sub["time"].astype(int), sub[column].astype(float)))
    return np.asarray([values.get(frame, np.nan) for frame in range(start, end + 1)])


def plot_contour_centers(
    center_df: pd.DataFrame,
    row: pd.Series,
    start: int,
    end: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    handles = []
    labels = []
    frames = np.arange(start, end + 1)

    for method in TRAJECTORY_METHOD_ORDER:
        sub = center_df[center_df["method"] == method].sort_values("time")
        if sub.empty:
            continue
        style = dict(TRAJECTORY_LINE_STYLES[method])
        style.update(FIGURE_LINE_STYLE_OVERRIDES.get(method, {}))
        line, = ax.plot(
            complete_frame_series(sub, start, end, "center_x"),
            complete_frame_series(sub, start, end, "center_y"),
            color=FIGURE_METHOD_COLORS[method],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker="o",
            markersize=max(2.2, float(style["markersize"])),
            alpha=style.get("alpha", 0.55),
            zorder=style["zorder"],
            label=method,
        )
        handles.append(line)
        labels.append(method)

        # Place one small arrowhead at the midpoint of every consecutive pair;
        # no line or arrow shaft is drawn behind it.
        arrow_rows = sub[["time", "center_x", "center_y"]].to_numpy(dtype=float)
        arrow_marker_size = (1.4 * max(2.2, float(style["markersize"]))) ** 2
        for point0, point1 in zip(arrow_rows[:-1], arrow_rows[1:]):
            if not np.isfinite(point0).all() or not np.isfinite(point1).all():
                continue
            if int(point1[0]) != int(point0[0]) + 1:
                continue
            delta = point1[1:] - point0[1:]
            if float(np.linalg.norm(delta)) <= 1e-9:
                continue
            midpoint = 0.5 * (point0[1:] + point1[1:])
            screen_angle = float(np.degrees(np.arctan2(-delta[1], delta[0])))
            ax.scatter(
                [midpoint[0]],
                [midpoint[1]],
                marker=(3, 0, screen_angle - 90.0),
                s=arrow_marker_size,
                color=FIGURE_METHOD_COLORS[method],
                linewidths=0,
                alpha=0.8,
                zorder=style["zorder"] + 0.5,
            )

        # A hollow point marks each missing frame at its linearly interpolated
        # position without changing the trajectory data.
        observed = set(sub["time"].astype(int))
        for frame in frames:
            if frame in observed or frame == start or frame == end:
                continue
            previous = sub[sub["time"] < frame].tail(1)
            following = sub[sub["time"] > frame].head(1)
            if previous.empty or following.empty:
                continue
            left = previous.iloc[0]
            right = following.iloc[0]
            fraction = (frame - left["time"]) / (right["time"] - left["time"])
            x = left["center_x"] + fraction * (right["center_x"] - left["center_x"])
            y = left["center_y"] + fraction * (right["center_y"] - left["center_y"])
            ax.scatter(
                [x],
                [y],
                marker="o",
                s=20,
                facecolors="white",
                edgecolors=FIGURE_METHOD_COLORS[method],
                linewidths=0.8,
                alpha=0.8,
                zorder=style["zorder"] + 0.1,
            )

    ax.set_title(
        f"Contour center: GT {int(row['gt_tid'])} ({row['gt_role']})",
        fontweight="bold",
        loc="left",
    )
    ax.set_xlabel("Image X (pixel)")
    ax.set_ylabel("Image Y (pixel)")
    ax.set_aspect("equal", adjustable="box")
    x_values = center_df["center_x"].to_numpy(dtype=float)
    y_values = center_df["center_y"].to_numpy(dtype=float)
    image_width = int(center_df["image_width"].dropna().min())
    image_height = int(center_df["image_height"].dropna().min())
    x_min = max(0.0, float(np.nanmin(x_values)) - 30.0)
    x_max = min(float(image_width - 1), float(np.nanmax(x_values)) + 30.0)
    y_min = max(0.0, float(np.nanmin(y_values)) - 30.0)
    y_max = min(float(image_height - 1), float(np.nanmax(y_values)) + 30.0)
    axis_limits = CONTOUR_CENTER_AXIS_LIMITS.get(int(row["gt_tid"]), {})
    ax.set_xlim(*axis_limits.get("xlim", (x_min, x_max)))
    ax.set_ylim(*axis_limits.get("ylim", (y_max, y_min)))
    if "xticks" in axis_limits:
        ax.set_xticks(list(axis_limits["xticks"]))
    ax.grid(False)
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        labelspacing=0.35,
        handlelength=2.4,
    )
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def center_distance_rows(center_df: pd.DataFrame) -> pd.DataFrame:
    gt = center_df[center_df["method"] == "GT"][["time", "center_x", "center_y"]].rename(
        columns={"center_x": "gt_center_x", "center_y": "gt_center_y"}
    )
    rows = []
    for method in ("SORT", "LiveCellX"):
        merged = center_df[center_df["method"] == method].merge(gt, on="time", how="inner")
        for record in merged.itertuples(index=False):
            distance = np.hypot(
                record.center_x - record.gt_center_x,
                record.center_y - record.gt_center_y,
            )
            rows.append(
                {
                    "gt_tid": int(record.gt_tid),
                    "method": method,
                    "track_id": int(record.track_id),
                    "time": int(record.time),
                    "center_distance_to_gt_pixels": float(distance),
                }
            )
    return pd.DataFrame(rows)


def plot_center_distance_to_gt(
    distance_df: pd.DataFrame,
    case: Dict[str, int],
    output_path: Path,
) -> None:
    start = int(case["start_frame"])
    end = int(case["end_frame"])
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for method in ("SORT", "LiveCellX"):
        sub = distance_df[distance_df["method"] == method].sort_values("time")
        if sub.empty:
            continue
        style = dict(TRAJECTORY_LINE_STYLES[method])
        ax.plot(
            sub["time"],
            sub["center_distance_to_gt_pixels"],
            color=FIGURE_METHOD_COLORS[method],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=style["markersize"],
            alpha=style.get("alpha", 0.55),
            label=method,
        )
    ax.axvline(
        int(case["switch_frame"]),
        color="#6f6f6f",
        linewidth=0.7,
        linestyle="--",
        alpha=0.55,
    )
    ax.set_title(f"Center distance to GT: GT {case['gt_tid']}", fontweight="bold", loc="left")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Distance to GT center (pixels)")
    ax.set_xticks(list(range(start, end + 1)))
    ax.set_xlim(start, end)
    ax.set_ylim(bottom=0)
    ax.grid(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def standardized_feature_rows(feature_df: pd.DataFrame) -> tuple:
    standardized = feature_df.copy()
    correlation_rows = []
    for feature, _, _, _ in TRAJECTORY_FEATURE_SPECS:
        z_column = f"{feature}_zscore"
        standardized[z_column] = np.nan
        for method in TRAJECTORY_METHOD_ORDER:
            index = standardized.index[standardized["method"] == method]
            values = standardized.loc[index, feature].astype(float)
            valid = values[np.isfinite(values)]
            if valid.empty:
                continue
            std = float(valid.std(ddof=0))
            if std <= 0 or not np.isfinite(std):
                standardized.loc[index, z_column] = 0.0
            else:
                standardized.loc[index, z_column] = (values - float(valid.mean())) / std

        pivot = standardized.pivot(index="time", columns="method", values=z_column)
        for method in ("SORT", "LiveCellX"):
            pair = pivot[["GT", method]].dropna() if {"GT", method}.issubset(pivot.columns) else pd.DataFrame()
            correlation = float(pair["GT"].corr(pair[method])) if len(pair) >= 2 else np.nan
            correlation_rows.append(
                {
                    "feature": feature,
                    "method": method,
                    "pearson_r_vs_gt": correlation,
                    "n_frames": int(len(pair)),
                }
            )
    return standardized, pd.DataFrame(correlation_rows)


def _closed_contour_xy(sc, crop_origin: tuple) -> tuple:
    points = np.asarray(sc.contour, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return np.asarray([]), np.asarray([])
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    row0, col0 = crop_origin
    return points[:, 1] - col0, points[:, 0] - row0


def plot_keyframe_contour_overlay(
    case: Dict[str, int],
    gt_sctc,
    sort_sctc,
    livecellx_sctc,
    output_path: Path,
    padding: int = 20,
) -> None:
    # SORT is drawn last with a dashed line so it remains visible when its
    # contour is nearly identical to GT or LiveCellX.
    methods = (
        ("GT", gt_sctc, int(case["gt_tid"]), 2.8, "-", 1),
        ("LiveCellX", livecellx_sctc, int(case["livecellx_tid"]), 1.9, "-", 2),
        ("SORT", sort_sctc, int(case["sort_tid"]), 2.1, (0, (3, 2)), 3),
    )
    key_frames = [int(frame) for frame in case["key_frames"]]
    fig, axes = plt.subplots(1, len(key_frames), figsize=(7.8, 2.65))
    axes = np.atleast_1d(axes)
    for ax, frame in zip(axes, key_frames):
        cells = []
        for method, sctc, track_id, linewidth, linestyle, zorder in methods:
            trajectory = sctc.get_trajectory(track_id)
            sc = trajectory.timeframe_to_single_cell.get(frame)
            if sc is not None:
                cells.append((method, sc, linewidth, linestyle, zorder))
        if not cells:
            ax.text(0.5, 0.5, "No cells", ha="center", va="center", transform=ax.transAxes)
            continue

        all_points = np.vstack([np.asarray(sc.contour, dtype=float) for _, sc, _, _, _ in cells])
        image = cells[0][1].get_img()
        row0 = max(0, int(np.floor(np.nanmin(all_points[:, 0]))) - int(padding))
        row1 = min(image.shape[0], int(np.ceil(np.nanmax(all_points[:, 0]))) + int(padding) + 1)
        col0 = max(0, int(np.floor(np.nanmin(all_points[:, 1]))) - int(padding))
        col1 = min(image.shape[1], int(np.ceil(np.nanmax(all_points[:, 1]))) + int(padding) + 1)
        crop = np.asarray(image[row0:row1, col0:col1])
        finite = crop[np.isfinite(crop)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [1, 99])
        else:
            vmin, vmax = None, None
        ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax)
        for method, sc, linewidth, linestyle, zorder in cells:
            x, y = _closed_contour_xy(sc, (row0, col0))
            ax.plot(
                x,
                y,
                color=FIGURE_METHOD_COLORS[method],
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.95,
                zorder=zorder,
            )
        ax.set_title(f"Frame {frame}", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=FIGURE_METHOD_COLORS[method],
            linewidth=linewidth,
            linestyle=linestyle,
            label=method,
        )
        for method, _, _, linewidth, linestyle, _ in methods
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.88, 0.5), frameon=False)
    fig.suptitle(f"Cell identity through the tracking switch: GT {case['gt_tid']}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.87, 0.92))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_case_readme(case: Dict[str, int], output_dir: Path) -> None:
    text = f"""# Annotated trajectory case

- GT trajectory: `{case['gt_tid']}`
- SORT trajectory: `{case['sort_tid']}`
- LiveCellX trajectory: `{case['livecellx_tid']}`
- Frame span: `{case['start_frame']}-{case['end_frame']}` (inclusive)
- Identity-switch frame: `{case['switch_frame']}`
- Key frames: `{case['key_frames']}`
- IDs are fixed; no trajectory matching is performed by this plotting script.

## Outputs

- `area.png`, `perimeter.png`, `eccentricity.png`, `solidity.png`, and
  `aspect_ratio.png`: raw cell-shape features over time.
- `*_zscore.png`: within-trajectory standardized temporal profiles. Raw plots
  are retained so that standardization does not hide segmentation bias.
- `shape_pca.png`: all methods projected into one PCA basis fitted only on the
  edited GT contours after smoothing, cyclic landmark registration, and
  generalized Procrustes alignment.
- `contour_center.png`: contour-mask-centroid path in image coordinates.
- `center_distance_to_gt.png`: per-frame contour-center distance to GT.
- `keyframe_contour_overlay.png`: raw images before, at, and after the
  identity switch with GT, SORT, and LiveCellX contours.
- CSV files contain all plotted source values and temporal correlations.
"""
    (output_dir / "README.md").write_text(text)


def generate_case(
    case,
    gt_sctc,
    sort_sctc,
    livecellx_sctc,
    output_root: Path,
    exact_mask_area: bool,
    colorbar_min: float,
    colorbar_max: float,
    shape_df: pd.DataFrame,
    shape_xlim: tuple,
    shape_ylim: tuple,
) -> None:
    for method, sctc, key in (
        ("GT", gt_sctc, "gt_tid"),
        ("SORT", sort_sctc, "sort_tid"),
        ("LiveCellX", livecellx_sctc, "livecellx_tid"),
    ):
        require_track(sctc, int(case[key]), method)

    start = int(case["start_frame"])
    end = int(case["end_frame"])
    row = mapping_row(case)
    case_dir = output_root / trajectory_folder_name(int(case["gt_tid"]), "annotated_traj")
    case_dir.mkdir(parents=True, exist_ok=True)

    feature_df = feature_rows(case, gt_sctc, sort_sctc, livecellx_sctc, exact_mask_area)
    feature_df.to_csv(case_dir / "feature_series.csv", index=False)
    sort_times = feature_df.loc[feature_df["method"] == "SORT", "time"].astype(int)
    if sort_times.empty:
        raise ValueError(f"SORT trajectory {case['sort_tid']} has no cells in frames {start}-{end}")
    switch_frame = int(case["switch_frame"])
    for feature, title, ylabel, transform in TRAJECTORY_FEATURE_SPECS:
        # The general downstream script log-transforms area. For these two
        # illustrative cases, retain the actual pixel area so that mask-size
        # differences remain directly visible.
        if feature == "area":
            ylabel = "Area (pixels)"
            transform = None
        plot_single_trajectory_feature(
            feature_df,
            row,
            feature,
            title,
            ylabel,
            transform,
            case_dir,
            methods=TRAJECTORY_METHOD_ORDER,
            figsize=(4.2, 3.0),
            vertical_line_time=switch_frame,
            xticks=list(range(start, end + 1)),
            line_style_overrides=FIGURE_LINE_STYLE_OVERRIDES,
            method_color_overrides=FIGURE_METHOD_COLORS,
        )

    standardized_df, correlation_df = standardized_feature_rows(feature_df)
    standardized_df.to_csv(case_dir / "standardized_feature_series.csv", index=False)
    correlation_df.insert(0, "gt_tid", int(case["gt_tid"]))
    correlation_df.to_csv(case_dir / "feature_temporal_correlations.csv", index=False)
    for feature, title, _, _ in TRAJECTORY_FEATURE_SPECS:
        z_column = f"{feature}_zscore"
        plot_single_trajectory_feature(
            standardized_df,
            row,
            z_column,
            f"Standardized {title}",
            "Within-trajectory z score",
            None,
            case_dir,
            methods=TRAJECTORY_METHOD_ORDER,
            figsize=(4.2, 3.0),
            vertical_line_time=switch_frame,
            xticks=list(range(start, end + 1)),
            line_style_overrides=FIGURE_LINE_STYLE_OVERRIDES,
            method_color_overrides=FIGURE_METHOD_COLORS,
        )

    shape_df.to_csv(case_dir / "shape_pca_series.csv", index=False)
    plot_gt_reference_shape_pca(
        shape_df,
        case,
        case_dir / "shape_pca.png",
        colorbar_min,
        colorbar_max,
        shape_xlim,
        shape_ylim,
    )
    # PC distance is deliberately not reported as an identity metric: nearby
    # cells can have similar outlines even when tracking identity is wrong.
    for obsolete_name in ("shape_distance_to_gt.png", "shape_distance_to_gt_series.csv"):
        obsolete_path = case_dir / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()

    center_df = contour_center_rows(case, gt_sctc, sort_sctc, livecellx_sctc)
    center_df.to_csv(case_dir / "contour_center_series.csv", index=False)
    plot_contour_centers(center_df, row, start, end, case_dir / "contour_center.png")
    center_distance_df = center_distance_rows(center_df)
    center_distance_df.to_csv(case_dir / "center_distance_to_gt_series.csv", index=False)
    plot_center_distance_to_gt(center_distance_df, case, case_dir / "center_distance_to_gt.png")
    plot_keyframe_contour_overlay(
        case,
        gt_sctc,
        sort_sctc,
        livecellx_sctc,
        case_dir / "keyframe_contour_overlay.png",
    )
    for obsolete_name in ("bbox_center.png", "bbox_center_series.csv"):
        obsolete_path = case_dir / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()
    write_case_readme(case, case_dir)
    print(f"[done] GT {case['gt_tid']}: {case_dir}", flush=True)


def run(args: argparse.Namespace) -> None:
    gt_path = require_file(Path(args.gt_path), "edited GT collection")
    sort_path = require_file(Path(args.sort_path), "SORT collection")
    livecellx_path = require_file(Path(args.livecellx_path), "LiveCellX collection")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load:GT] {gt_path}", flush=True)
    gt_sctc = load_sctc(gt_path)
    print(f"[load:SORT] {sort_path}", flush=True)
    sort_sctc = load_sctc(sort_path)
    print(f"[load:LiveCellX] {livecellx_path}", flush=True)
    livecellx_sctc = load_sctc(livecellx_path)

    print("[shape-model] fitting the shared PCA basis from edited GT contours", flush=True)
    shape_model = build_gt_reference_shape_model(
        gt_sctc,
        contour_points=int(args.shape_pca_contour_points),
    )
    case_shape_dfs = {
        int(case["gt_tid"]): project_case_shapes(
            case,
            gt_sctc,
            sort_sctc,
            livecellx_sctc,
            shape_model,
        )
        for case in CASES
    }
    all_shape_df = pd.concat(case_shape_dfs.values(), ignore_index=True)
    if all_shape_df.empty:
        raise ValueError("No valid contours were available for shape projection")
    pc1 = all_shape_df["shape_pc1"].to_numpy(dtype=float)
    pc2 = all_shape_df["shape_pc2"].to_numpy(dtype=float)
    pc1_pad = max(0.05, 0.08 * float(np.nanmax(pc1) - np.nanmin(pc1)))
    pc2_pad = max(0.05, 0.08 * float(np.nanmax(pc2) - np.nanmin(pc2)))
    shape_xlim = (float(np.nanmin(pc1) - pc1_pad), float(np.nanmax(pc1) + pc1_pad))
    shape_ylim = (float(np.nanmin(pc2) - pc2_pad), float(np.nanmax(pc2) + pc2_pad))

    model_metadata = {
        "description": "GT-reference active-shape point-distribution model",
        "training_source": "all valid contours in the edited GT collection",
        "gt_source_track_frames": [list(value) for value in shape_model["gt_sources"]],
        "n_gt_shapes": len(shape_model["gt_sources"]),
        "contour_points": int(shape_model["contour_points"]),
        "smoothing": "two periodic [1, 2, 1] / 4 passes",
        "alignment": "orientation normalization, cyclic landmark registration, generalized Procrustes without reflection",
        "pca_explained_variance_ratio": shape_model["pca"].explained_variance_ratio_.tolist(),
        "pca_mean": shape_model["pca"].mean_.tolist(),
        "pca_components": shape_model["pca"].components_.tolist(),
        "procrustes_reference_shape": shape_model["reference"].tolist(),
        "shared_pc1_limits": list(shape_xlim),
        "shared_pc2_limits": list(shape_ylim),
    }
    (output_dir / "gt_reference_shape_model.json").write_text(json.dumps(model_metadata, indent=2))

    # One absolute-frame color scale and one pair of PC limits are shared by
    # both shape-PCA figures.
    colorbar_min = float(min(case["start_frame"] for case in CASES))
    colorbar_max = float(max(case["end_frame"] for case in CASES))
    for case in CASES:
        generate_case(
            case,
            gt_sctc,
            sort_sctc,
            livecellx_sctc,
            output_dir,
            exact_mask_area=bool(args.exact_mask_area),
            colorbar_min=colorbar_min,
            colorbar_max=colorbar_max,
            shape_df=case_shape_dfs[int(case["gt_tid"])],
            shape_xlim=shape_xlim,
            shape_ylim=shape_ylim,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-path", default=str(DEFAULT_GT_PATH))
    parser.add_argument("--sort-path", default=str(DEFAULT_SORT_PATH))
    parser.add_argument("--livecellx-path", default=str(DEFAULT_LIVECELLX_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--shape-pca-contour-points", type=int, default=128)
    parser.add_argument("--exact-mask-area", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
