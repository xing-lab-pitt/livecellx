#!/usr/bin/env python
"""Five trajectory metrics on the annotated process2 GT subgraph.

This script implements exactly the five metrics requested for the rebuttal:

    ATR, ATP, ATA, AssA, and BC(5)

All metrics are computed from:

    1. the manually annotated GT trajectory collection, and
    2. each method trajectory collection already restricted to the GT region.

The script does not use all-cell method collections. Same-frame mask matching is
used to construct accepted GT-prediction detections. ATR/ATP/ATA are computed
from temporal track IoU with one global Hungarian trajectory assignment. AssA is
the HOTA-style association accuracy over accepted detections. BC(5) requires
explicit predicted mother-daughter links and is reported as unavailable when a
method collection does not contain predicted division links.
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-livecellx")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from livecellx.core.single_cell import SingleCellTrajectory, SingleCellTrajectoryCollection
from process2_ctc_style_tracking_evaluation import (
    FINAL_TRAJ_COLLECTION_DIR,
    GT_PATH,
    LATEST_RESULTS_DIR,
    MatchConfig,
    build_gt_cell_table,
    lineage_edges_from_gt,
    load_sctc,
    match_sctc_condition,
    relation_id_set,
    relation_ids_from_gt,
    resolve_path,
)


OUT_DIR = LATEST_RESULTS_DIR / "trajectory_level_metrics"
SORT_BEFORE_GT_REGION_PATH = FINAL_TRAJ_COLLECTION_DIR / "sctc-final-2026-0610_before_livecellx_gt_region_only.json"
SORT_AFTER_GT_REGION_PATH = (
    FINAL_TRAJ_COLLECTION_DIR
    / "sctc-final-2026-06-02_before_correction_timesformer_lineage_corrected_gt_region_only.json"
)
ULTRACK_GT_REGION_PATH = FINAL_TRAJ_COLLECTION_DIR / "ultrack_from_cellpose_process2_beforecsnet_exp4_gt_region_only.json"

METHODS = [
    ("SORT", "#3F6DB5"),
    ("LiveCellX", "#F28E2B"),
    ("Ultrack", "#009E73"),
]
METHOD_ORDER = [x[0] for x in METHODS]
METHOD_COLORS = {x[0]: x[1] for x in METHODS}

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


def save_figure(fig: plt.Figure, out_base: Path, formats: Sequence[str], dpi: int = 450) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=dpi if fmt.lower() in {"png", "tif", "tiff"} else None)
    plt.close(fig)


def nonempty_track_ids(sctc: SingleCellTrajectoryCollection) -> List[int]:
    return sorted(int(tid) for tid, traj in sctc if len(traj.times) > 0)


def gt_track_lengths(gt_sctc: SingleCellTrajectoryCollection, gt_ids: Set[int]) -> Dict[int, int]:
    lengths: Dict[int, int] = {}
    for tid in sorted(int(x) for x in gt_ids):
        if tid in gt_sctc.track_id_to_trajectory:
            lengths[tid] = int(len(gt_sctc.get_trajectory(tid).times))
    return lengths


def predicted_track_lengths(
    target_sctc: SingleCellTrajectoryCollection,
    min_frame: int = 0,
    max_frame: Optional[int] = None,
) -> Dict[int, int]:
    lengths: Dict[int, int] = {}
    for tid, traj in target_sctc:
        times = [int(t) for t in traj.times if int(t) >= int(min_frame)]
        if max_frame is not None:
            times = [t for t in times if t <= int(max_frame)]
        if times:
            lengths[int(tid)] = int(len(set(times)))
    return lengths


def accepted_matches(match_df: pd.DataFrame) -> pd.DataFrame:
    if match_df.empty:
        return match_df.copy()
    return match_df[match_df["matched"].fillna(False).astype(bool) & match_df["target_tid"].notna()].copy()


def pair_match_counts(match_df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    tp = accepted_matches(match_df)
    counts: Dict[Tuple[int, int], int] = {}
    if tp.empty:
        return counts
    grouped = tp.groupby(["gt_tid", "target_tid"]).size().reset_index(name="matched_frame_count")
    for _, row in grouped.iterrows():
        counts[(int(row["gt_tid"]), int(row["target_tid"]))] = int(row["matched_frame_count"])
    return counts


def compute_temporal_iou_assignment(
    condition: str,
    match_df: pd.DataFrame,
    gt_lengths: Dict[int, int],
    pred_lengths: Dict[int, int],
) -> Tuple[dict, pd.DataFrame, Dict[int, Optional[int]], Dict[int, Optional[int]]]:
    gt_ids = sorted(int(x) for x in gt_lengths)
    pred_ids = sorted(int(x) for x in pred_lengths)
    counts = pair_match_counts(match_df)

    q = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
    for i, gt_tid in enumerate(gt_ids):
        n_i = int(gt_lengths[gt_tid])
        for j, pred_tid in enumerate(pred_ids):
            c_ij = int(counts.get((gt_tid, pred_tid), 0))
            m_j = int(pred_lengths[pred_tid])
            denom = n_i + m_j - c_ij
            q[i, j] = float(c_ij / denom) if denom > 0 else 0.0

    assigned_pairs: Set[Tuple[int, int]] = set()
    s_score = 0.0
    gt_to_pred: Dict[int, Optional[int]] = {gt_tid: None for gt_tid in gt_ids}
    pred_to_gt: Dict[int, Optional[int]] = {pred_tid: None for pred_tid in pred_ids}
    if q.size and len(gt_ids) and len(pred_ids):
        row_ind, col_ind = linear_sum_assignment(-q)
        for r, c in zip(row_ind, col_ind):
            score = float(q[r, c])
            if score <= 0:
                continue
            gt_tid = int(gt_ids[r])
            pred_tid = int(pred_ids[c])
            assigned_pairs.add((gt_tid, pred_tid))
            gt_to_pred[gt_tid] = pred_tid
            pred_to_gt[pred_tid] = gt_tid
            s_score += score

    k = len(gt_ids)
    k_hat = len(pred_ids)
    atr = float(s_score / k) if k else np.nan
    atp = float(s_score / k_hat) if k_hat else np.nan
    ata = float(2.0 * s_score / (k + k_hat)) if (k + k_hat) else np.nan

    debug_rows = []
    for gt_tid in gt_ids:
        for pred_tid in pred_ids:
            c_ij = int(counts.get((gt_tid, pred_tid), 0))
            n_i = int(gt_lengths[gt_tid])
            m_j = int(pred_lengths[pred_tid])
            denom = n_i + m_j - c_ij
            q_ij = float(c_ij / denom) if denom > 0 else 0.0
            if c_ij == 0 and (gt_tid, pred_tid) not in assigned_pairs:
                continue
            debug_rows.append(
                {
                    "condition": condition,
                    "gt_track_id": int(gt_tid),
                    "pred_track_id": int(pred_tid),
                    "gt_track_length": int(n_i),
                    "pred_track_length_in_gt_region": int(m_j),
                    "matched_frame_count": int(c_ij),
                    "track_iou": float(q_ij),
                    "globally_assigned": bool((gt_tid, pred_tid) in assigned_pairs),
                }
            )

    summary = {
        "condition": condition,
        "ATR": atr,
        "ATP": atp,
        "ATA": ata,
    }
    return summary, pd.DataFrame(debug_rows), gt_to_pred, pred_to_gt


def compute_assa(match_df: pd.DataFrame) -> float:
    tp = accepted_matches(match_df)
    if tp.empty:
        return np.nan
    counts = pair_match_counts(match_df)
    gt_totals: Dict[int, int] = {}
    pred_totals: Dict[int, int] = {}
    for (gt_tid, pred_tid), c_ij in counts.items():
        gt_totals[gt_tid] = gt_totals.get(gt_tid, 0) + int(c_ij)
        pred_totals[pred_tid] = pred_totals.get(pred_tid, 0) + int(c_ij)

    numerator = 0.0
    denominator = 0.0
    for (gt_tid, pred_tid), c_ij in counts.items():
        tpa = float(c_ij)
        fna = float(gt_totals.get(gt_tid, 0) - c_ij)
        fpa = float(pred_totals.get(pred_tid, 0) - c_ij)
        assoc = tpa / (tpa + fna + fpa) if (tpa + fna + fpa) > 0 else np.nan
        if np.isfinite(assoc):
            numerator += float(c_ij) * assoc
            denominator += float(c_ij)
    return float(numerator / denominator) if denominator > 0 else np.nan


def division_events_from_gt(edges_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if edges_df.empty:
        return pd.DataFrame()
    for mother_id, group in edges_df.groupby("mother_gt_tid"):
        daughters = sorted(int(x) for x in group["daughter_gt_tid"].tolist())
        if len(daughters) < 2:
            continue
        d1, d2 = daughters[:2]
        daughter_starts = {
            int(row["daughter_gt_tid"]): int(row["daughter_start"])
            for _, row in group.iterrows()
        }
        rows.append(
            {
                "gt_mother_id": int(mother_id),
                "gt_daughter_1_id": int(d1),
                "gt_daughter_2_id": int(d2),
                "gt_mother_start": int(group["mother_start"].iloc[0]),
                "gt_mother_end": int(group["mother_end"].iloc[0]),
                "gt_daughter_1_start": int(daughter_starts[d1]),
                "gt_daughter_2_start": int(daughter_starts[d2]),
                "gt_division_frame": int(max(daughter_starts[d1], daughter_starts[d2])),
            }
        )
    return pd.DataFrame(rows)


def predicted_division_events(target_sctc: SingleCellTrajectoryCollection) -> pd.DataFrame:
    rows = []
    for mother_id, mother in target_sctc:
        mother_id = int(mother_id)
        daughter_ids = relation_id_set(
            mother,
            "daughter_trajectories",
            SingleCellTrajectory.META_DAUGHTER_IDS,
        )
        daughter_ids = sorted(int(x) for x in daughter_ids if int(x) in target_sctc.track_id_to_trajectory)
        if len(daughter_ids) < 2:
            continue
        for d1, d2 in combinations(daughter_ids, 2):
            if len({mother_id, int(d1), int(d2)}) < 3:
                continue
            d1_traj = target_sctc.get_trajectory(int(d1))
            d2_traj = target_sctc.get_trajectory(int(d2))
            if not d1_traj.times or not d2_traj.times:
                continue
            pred_t = int(max(min(int(t) for t in d1_traj.times), min(int(t) for t in d2_traj.times)))
            rows.append(
                {
                    "pred_event_id": f"{mother_id}->{d1},{d2}",
                    "pred_mother_id": int(mother_id),
                    "pred_daughter_1_id": int(d1),
                    "pred_daughter_2_id": int(d2),
                    "pred_division_frame": int(pred_t),
                }
            )
    return pd.DataFrame(rows)


def compute_bc5(
    condition: str,
    gt_events: pd.DataFrame,
    pred_events: pd.DataFrame,
    gt_to_pred: Dict[int, Optional[int]],
    tolerance: int = 5,
) -> Tuple[float, pd.DataFrame]:
    if gt_events.empty:
        return np.nan, pd.DataFrame()
    if pred_events.empty:
        rows = []
        for _, ev in gt_events.iterrows():
            rows.append(
                {
                    "condition": condition,
                    "gt_mother_id": int(ev["gt_mother_id"]),
                    "gt_daughter_1_id": int(ev["gt_daughter_1_id"]),
                    "gt_daughter_2_id": int(ev["gt_daughter_2_id"]),
                    "pred_mother_id": np.nan,
                    "pred_daughter_1_id": np.nan,
                    "pred_daughter_2_id": np.nan,
                    "gt_division_frame": int(ev["gt_division_frame"]),
                    "pred_division_frame": np.nan,
                    "division_time_difference": np.nan,
                    "valid_parent_links": False,
                    "three_distinct_predicted_ids": False,
                    "matched_within_5_frames": False,
                }
            )
        return np.nan, pd.DataFrame(rows)

    candidates = []
    for gi, ev in gt_events.reset_index(drop=True).iterrows():
        p_m = gt_to_pred.get(int(ev["gt_mother_id"]))
        p_d1 = gt_to_pred.get(int(ev["gt_daughter_1_id"]))
        p_d2 = gt_to_pred.get(int(ev["gt_daughter_2_id"]))
        expected = {p_m, p_d1, p_d2}
        has_three = None not in expected and len({int(x) for x in expected if x is not None}) == 3
        for pi, pev in pred_events.reset_index(drop=True).iterrows():
            daughters_match = {p_d1, p_d2} == {
                int(pev["pred_daughter_1_id"]),
                int(pev["pred_daughter_2_id"]),
            }
            parent_match = p_m == int(pev["pred_mother_id"])
            dt = int(pev["pred_division_frame"]) - int(ev["gt_division_frame"])
            valid = bool(has_three and parent_match and daughters_match and abs(dt) <= int(tolerance))
            if valid:
                candidates.append((gi, pi, abs(dt), dt))

    matched_gt: Set[int] = set()
    matched_pred: Set[int] = set()
    for gi, pi, _, _ in sorted(candidates, key=lambda x: (x[2], x[0], x[1])):
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(int(gi))
        matched_pred.add(int(pi))

    rows = []
    pred_events_reset = pred_events.reset_index(drop=True)
    for gi, ev in gt_events.reset_index(drop=True).iterrows():
        p_m = gt_to_pred.get(int(ev["gt_mother_id"]))
        p_d1 = gt_to_pred.get(int(ev["gt_daughter_1_id"]))
        p_d2 = gt_to_pred.get(int(ev["gt_daughter_2_id"]))
        best_pi = None
        best_abs_dt = np.inf
        for pi, pev in pred_events_reset.iterrows():
            parent_match = p_m == int(pev["pred_mother_id"])
            daughters_match = {p_d1, p_d2} == {
                int(pev["pred_daughter_1_id"]),
                int(pev["pred_daughter_2_id"]),
            }
            dt = int(pev["pred_division_frame"]) - int(ev["gt_division_frame"])
            if parent_match and daughters_match and abs(dt) < best_abs_dt:
                best_pi = int(pi)
                best_abs_dt = abs(dt)
        if best_pi is not None:
            pev = pred_events_reset.iloc[best_pi]
            dt = int(pev["pred_division_frame"]) - int(ev["gt_division_frame"])
            pred_m = int(pev["pred_mother_id"])
            pred_d1 = int(pev["pred_daughter_1_id"])
            pred_d2 = int(pev["pred_daughter_2_id"])
            three_distinct = len({pred_m, pred_d1, pred_d2}) == 3
            matched = int(gi) in matched_gt and best_pi in matched_pred
        else:
            dt = np.nan
            pred_m = pred_d1 = pred_d2 = np.nan
            three_distinct = False
            matched = False
        rows.append(
            {
                "condition": condition,
                "gt_mother_id": int(ev["gt_mother_id"]),
                "gt_daughter_1_id": int(ev["gt_daughter_1_id"]),
                "gt_daughter_2_id": int(ev["gt_daughter_2_id"]),
                "pred_mother_id": pred_m,
                "pred_daughter_1_id": pred_d1,
                "pred_daughter_2_id": pred_d2,
                "gt_division_frame": int(ev["gt_division_frame"]),
                "pred_division_frame": np.nan if best_pi is None else int(pred_events_reset.iloc[best_pi]["pred_division_frame"]),
                "division_time_difference": dt,
                "valid_parent_links": bool(best_pi is not None),
                "three_distinct_predicted_ids": bool(three_distinct),
                "matched_within_5_frames": bool(matched),
            }
        )

    tp_b = float(len(matched_gt))
    fn_b = float(len(gt_events) - len(matched_gt))
    fp_b = float(len(pred_events) - len(matched_pred))
    bc5 = float(2.0 * tp_b / (2.0 * tp_b + fp_b + fn_b)) if (2.0 * tp_b + fp_b + fn_b) > 0 else np.nan
    return bc5, pd.DataFrame(rows)

def plot_metric_summary(summary_df: pd.DataFrame, out_dir: Path, formats: Sequence[str]) -> None:
    metrics = ["ATR", "ATP", "ATA", "AssA", "BC(5)"]

    def plot_conditions(conditions: Sequence[str], output_stem: str) -> None:
        fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)
        x = np.arange(len(metrics))
        width = 0.22 if len(conditions) >= 3 else 0.28
        offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2.0) * width

        for idx, condition in enumerate(conditions):
            sub = summary_df[summary_df["condition"] == condition]
            vals = [
                float(sub[m].iloc[0]) * 100.0
                if len(sub) and pd.notna(sub[m].iloc[0])
                else np.nan
                for m in metrics
            ]
            bars = ax.bar(
                x + offsets[idx],
                vals,
                width=width,
                color=METHOD_COLORS[condition],
                edgecolor="white",
                linewidth=0.6,
                label=condition,
            )
            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 1.1,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        2.0,
                        "NA",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                        color="#4b5563",
                    )

        ax.set_title("Process2 annotated-GT subgraph tracking metrics", fontweight="bold", loc="left")
        ax.set_ylabel("Score (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 112)
        ax.grid(False)
        ax.legend(loc="upper right", ncol=1, frameon=False)
        save_figure(fig, out_dir / output_stem, formats)

    plot_conditions(METHOD_ORDER, "fig_process2_five_tracking_metrics")
    plot_conditions(
        ["SORT", "LiveCellX"],
        "fig_process2_five_tracking_metrics_sort_livecellx",
    )
    plot_conditions(
        ["LiveCellX", "Ultrack"],
        "fig_process2_five_tracking_metrics_livecellx_ultrack",
    )


def write_readme(out_dir: Path, args: argparse.Namespace, summary_df: pd.DataFrame) -> None:
    lines = [
        "# Process2 five tracking metrics on annotated GT subgraph",
        "",
        "This folder reports exactly five metrics: `ATR`, `ATP`, `ATA`, `AssA`, and `BC(5)`.",
        "",
        "Inputs:",
        "",
        f"- GT trajectory collection: `{resolve_path(Path(args.gt_path))}`",
        f"- SORT GT-region collection: `{resolve_path(Path(args.before_path))}`",
        f"- LiveCellX GT-region collection: `{resolve_path(Path(args.after_path))}`",
        f"- Ultrack GT-region collection: `{resolve_path(Path(args.ultrack_path))}`",
        "",
        "Rules:",
        "",
        f"- Evaluation frame window: `{args.min_frame}` to `{args.max_frame}` inclusive.",
        f"- Same-frame mask matching with IoU >= `{args.iou_threshold}`.",
        "- One-to-one Hungarian cell matching is performed independently in each frame.",
        "- ATR, ATP, and ATA use temporal track IoU and one global Hungarian trajectory assignment.",
        "- AssA is computed as the HOTA-style association Jaccard averaged over accepted detections.",
        "- BC(5) requires explicit predicted parent-daughter links. If a method has no predicted division links, BC(5) is reported as unavailable.",
        "",
        "Summary:",
        "",
        "```",
        summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "```",
    ]
    (out_dir / "README_process2_five_tracking_metrics.md").write_text("\n".join(lines))




def gt_track_lengths_from_cell_table(gt_cell_df: pd.DataFrame) -> Dict[int, int]:
    if gt_cell_df.empty:
        return {}
    return {int(tid): int(len(group)) for tid, group in gt_cell_df.groupby("gt_tid")}

def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.lower().lstrip(".") for fmt in args.figure_formats]

    gt_sctc = load_sctc(resolve_path(Path(args.gt_path)))
    gt_ids = relation_ids_from_gt(gt_sctc)
    gt_cell_df = build_gt_cell_table(gt_sctc, gt_ids, min_frame=int(args.min_frame), max_frame=args.max_frame)
    gt_lengths = gt_track_lengths_from_cell_table(gt_cell_df)
    edges_df = lineage_edges_from_gt(gt_sctc, gt_ids, min_frame=int(args.min_frame), max_frame=args.max_frame)
    gt_events = division_events_from_gt(edges_df)
    print(f"[gt] {len(gt_lengths)} tracks, {len(gt_cell_df)} cells, {len(gt_events)} division events")

    cfg = MatchConfig(iou_threshold=float(args.iou_threshold), min_candidate_iou=float(args.min_candidate_iou))
    sources = [
        ("SORT", resolve_path(Path(args.before_path))),
        ("LiveCellX", resolve_path(Path(args.after_path))),
        ("Ultrack", resolve_path(Path(args.ultrack_path))),
    ]

    all_summary = []
    all_matches = []
    all_track_debug = []
    all_division_debug = []
    all_pred_divisions = []

    for condition, path in sources:
        print(f"[load:{condition}] {path}")
        target_sctc = load_sctc(path)
        match_df = match_sctc_condition(condition, target_sctc, gt_sctc, gt_cell_df, cfg)
        match_df["mask_iou_threshold"] = float(args.iou_threshold)
        pred_lengths = predicted_track_lengths(
            target_sctc,
            min_frame=int(args.min_frame),
            max_frame=args.max_frame,
        )

        summary, track_debug, gt_to_pred, _ = compute_temporal_iou_assignment(
            condition,
            match_df,
            gt_lengths,
            pred_lengths,
        )
        summary["AssA"] = compute_assa(match_df)

        pred_events = predicted_division_events(target_sctc)
        if len(pred_events):
            pred_events.insert(0, "condition", condition)
        bc5, division_debug = compute_bc5(condition, gt_events, pred_events, gt_to_pred, tolerance=5)
        summary["BC(5)"] = bc5
        summary["mask_iou_threshold"] = float(args.iou_threshold)

        all_summary.append(summary)
        all_matches.append(match_df)
        all_track_debug.append(track_debug)
        all_division_debug.append(division_debug)
        if len(pred_events):
            all_pred_divisions.append(pred_events)
        del target_sctc

    summary_df = pd.DataFrame(all_summary)
    summary_df = summary_df[["condition", "mask_iou_threshold", "ATR", "ATP", "ATA", "AssA", "BC(5)"]]
    summary_df["condition"] = pd.Categorical(summary_df["condition"], categories=METHOD_ORDER, ordered=True)
    summary_df = summary_df.sort_values("condition").reset_index(drop=True)

    match_all = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    track_debug_df = pd.concat(all_track_debug, ignore_index=True) if all_track_debug else pd.DataFrame()
    division_debug_df = pd.concat(all_division_debug, ignore_index=True) if all_division_debug else pd.DataFrame()
    pred_division_df = pd.concat(all_pred_divisions, ignore_index=True) if all_pred_divisions else pd.DataFrame()

    summary_df.to_csv(out_dir / "process2_five_tracking_metrics_summary.csv", index=False)
    accepted_matches(match_all).rename(
        columns={"time": "frame", "gt_tid": "gt_track_id", "target_tid": "pred_track_id", "iou": "mask_iou"}
    )[["condition", "frame", "gt_track_id", "pred_track_id", "mask_iou", "mask_iou_threshold"]].to_csv(
        out_dir / "process2_five_tracking_metrics_accepted_matches.csv",
        index=False,
    )
    track_debug_df.to_csv(out_dir / "process2_five_tracking_metrics_per_track_debug.csv", index=False)
    division_debug_df.to_csv(out_dir / "process2_five_tracking_metrics_per_division_debug.csv", index=False)
    pred_division_df.to_csv(out_dir / "process2_five_tracking_metrics_predicted_divisions_debug.csv", index=False)
    gt_cell_df.to_csv(out_dir / "process2_five_tracking_metrics_gt_cells.csv", index=False)
    edges_df.to_csv(out_dir / "process2_five_tracking_metrics_gt_edges.csv", index=False)

    plot_metric_summary(summary_df, out_dir, formats)
    write_readme(out_dir, args, summary_df)
    print(f"[done] wrote five tracking metrics to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-path", default=str(GT_PATH))
    parser.add_argument("--before-path", default=str(SORT_BEFORE_GT_REGION_PATH))
    parser.add_argument("--after-path", default=str(SORT_AFTER_GT_REGION_PATH))
    parser.add_argument("--ultrack-path", default=str(ULTRACK_GT_REGION_PATH))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--min-frame", type=int, default=0, help="First frame included in GT evaluation.")
    parser.add_argument("--max-frame", type=int, default=99, help="Last frame included in GT evaluation; default keeps frames 0-99.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-candidate-iou", type=float, default=0.05)
    parser.add_argument("--figure-formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
