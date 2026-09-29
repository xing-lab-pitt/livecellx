#!/usr/bin/env python
"""
Generate a process2-only CSNet cell match rate figure.

Process2 is an application movie without pixel-level manually segmented masks.
The tracking-correction workflow supplies the reference used here: TD-1 detects
an under-segmentation when one mask in a target frame substantially overlaps
multiple individual masks in an adjacent reference frame (IoMin > 0.5).  For
target objects that CSNet splits, the neighboring individual masks are treated
as the reference cells, and cell match rate follows the manuscript equation:
the per-case fraction of reference cells matched to target-frame predicted
masks by IoU.

Outputs:
    results_process2_csnet/process2_csn_cell_match_temporal_reference.csv
    results_process2_csnet/process2_csn_cell_match_case_summary.csv
    latest_results/figures/fig3i_process2_csn_cell_match_rate.{pdf,png}
    latest_results/figures/fig3i_cell_match_iou_thresholds.{pdf,png}
"""

import csv
import os
import re
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-livecellx")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_MASK_DIR = SCRIPT_DIR / "data" / "process2" / "2D_mask"
CORRECTED_MASK_DIR = SCRIPT_DIR / "results_process2_csnet" / "livecellx_corrected_masks"
OUTPUT_DIR = SCRIPT_DIR / "results_process2_csnet"
FIG_DIR = SCRIPT_DIR / "latest_results" / "figures"
REBUTTAL_FIG_DIR = FIG_DIR
RATE_CSV_PATH = OUTPUT_DIR / "process2_csn_cell_match_temporal_reference.csv"
CASE_CSV_PATH = OUTPUT_DIR / "process2_csn_cell_match_case_summary.csv"

IOMIN_CANDIDATE_THRESHOLD = 0.5
IOU_THRESHOLDS = (0.5, 0.6, 0.7)


def frame_number(path):
    return int(re.findall(r"\d+", path.stem)[-1])


def sorted_masks(mask_dir, pattern):
    paths = sorted(mask_dir.glob(pattern), key=frame_number)
    if not paths:
        raise FileNotFoundError(f"No masks found in {mask_dir} with {pattern}")
    return paths


@lru_cache(maxsize=6)
def load_mask(path):
    return np.asarray(Image.open(path))


def object_areas(mask):
    labels, counts = np.unique(mask[mask > 0], return_counts=True)
    return dict(zip(labels.tolist(), counts.tolist()))


def overlap_tuples(first, second):
    """Return (first_label, second_label, intersection) for overlapping cells."""
    overlap = (first > 0) & (second > 0)
    if not np.any(overlap):
        return []
    encoded = (
        first[overlap].astype(np.uint64) << np.uint64(32)
    ) | second[overlap].astype(np.uint64)
    keys, counts = np.unique(encoded, return_counts=True)
    return [
        (int(key >> np.uint64(32)), int(key & np.uint64(0xFFFFFFFF)), int(count))
        for key, count in zip(keys, counts)
    ]


def substantial_references(reference_mask, target_mask, threshold=IOMIN_CANDIDATE_THRESHOLD):
    """Return target label -> substantially overlapping reference labels."""
    reference_areas = object_areas(reference_mask)
    target_areas = object_areas(target_mask)
    target_to_refs = {}
    target_to_scores = {}
    for reference_label, target_label, intersection in overlap_tuples(reference_mask, target_mask):
        iomin = intersection / min(reference_areas[reference_label], target_areas[target_label])
        if iomin > threshold:
            target_to_refs.setdefault(target_label, []).append(reference_label)
            target_to_scores.setdefault(target_label, []).append(iomin)
    return target_to_refs, target_to_scores


def find_td1_candidates(before_paths):
    """Find temporal under-segmentation candidates from the original masks."""
    candidates = {}
    for target_index, target_path in enumerate(before_paths):
        target_mask = load_mask(target_path)
        adjacent_indices = []
        if target_index > 0:
            adjacent_indices.append(target_index - 1)
        if target_index + 1 < len(before_paths):
            adjacent_indices.append(target_index + 1)
        for reference_index in adjacent_indices:
            refs, scores = substantial_references(load_mask(before_paths[reference_index]), target_mask)
            for target_label, reference_labels in refs.items():
                if len(reference_labels) < 2:
                    continue
                candidate = {
                    "target_index": target_index,
                    "target_label": target_label,
                    "reference_index": reference_index,
                    "reference_labels": reference_labels,
                    "selection_score": float(sum(scores[target_label])),
                }
                key = (target_index, target_label)
                previous = candidates.get(key)
                if previous is None or (
                    len(reference_labels), candidate["selection_score"]
                ) > (
                    len(previous["reference_labels"]), previous["selection_score"]
                ):
                    candidates[key] = candidate
    return list(candidates.values())


def corrected_split_count(candidate, before_mask, after_mask):
    """Count corrected components substantially overlapping a TD-1 target object."""
    target_binary = np.where(before_mask == candidate["target_label"], 1, 0).astype(np.uint32)
    after_to_original, _ = substantial_references(target_binary, after_mask)
    return sum(1 for original_labels in after_to_original.values() if 1 in original_labels)


def select_csn_split_candidates(candidates, before_paths, after_paths):
    selected = []
    for candidate in candidates:
        target_index = candidate["target_index"]
        split_count = corrected_split_count(
            candidate, load_mask(before_paths[target_index]), load_mask(after_paths[target_index])
        )
        if split_count >= 2:
            selected.append({**candidate, "corrected_component_count": split_count})
    return selected


def case_best_matches(reference_mask, prediction_mask, reference_labels):
    reference_areas = object_areas(reference_mask)
    prediction_areas = object_areas(prediction_mask)
    ref_position = {label: index for index, label in enumerate(reference_labels)}
    prediction_labels = set()
    overlaps = []
    for reference_label, prediction_label, intersection in overlap_tuples(reference_mask, prediction_mask):
        if reference_label not in ref_position:
            continue
        prediction_labels.add(prediction_label)
        overlaps.append((reference_label, prediction_label, intersection))

    if not prediction_labels:
        return np.zeros(len(reference_labels), dtype=float)

    prediction_labels = sorted(prediction_labels)
    pred_position = {label: index for index, label in enumerate(prediction_labels)}
    iou_matrix = np.zeros((len(reference_labels), len(prediction_labels)), dtype=float)
    for reference_label, prediction_label, intersection in overlaps:
        union = reference_areas[reference_label] + prediction_areas[prediction_label] - intersection
        iou_matrix[ref_position[reference_label], pred_position[prediction_label]] = intersection / union

    row_indices, column_indices = linear_sum_assignment(-iou_matrix)
    best_iou = np.zeros(len(reference_labels), dtype=float)
    for row, column in zip(row_indices, column_indices):
        best_iou[row] = iou_matrix[row, column]
    return best_iou


def evaluate_candidates(candidates, before_paths, after_paths):
    case_rows = []
    for case_id, candidate in enumerate(candidates, start=1):
        reference_mask = load_mask(before_paths[candidate["reference_index"]])
        before_best_iou = case_best_matches(
            reference_mask,
            load_mask(before_paths[candidate["target_index"]]),
            candidate["reference_labels"],
        )
        after_best_iou = case_best_matches(
            reference_mask,
            load_mask(after_paths[candidate["target_index"]]),
            candidate["reference_labels"],
        )
        for condition, values in (
            ("SORT", before_best_iou),
            ("LiveCellX", after_best_iou),
        ):
            for threshold in IOU_THRESHOLDS:
                matched = int(np.sum(values >= threshold))
                case_rows.append(
                    {
                        "case_id": case_id,
                        "condition": condition,
                        "iou_threshold": threshold,
                        "matched_cells": matched,
                        "reference_cells": len(candidate["reference_labels"]),
                        "cell_match_rate": matched / len(candidate["reference_labels"]),
                        "target_frame": candidate["target_index"] + 1,
                        "reference_frame": candidate["reference_index"] + 1,
                        "target_label": candidate["target_label"],
                        "corrected_component_count": candidate["corrected_component_count"],
                    }
                )
    return case_rows


def aggregate_rows(case_rows):
    rows = []
    for condition in ("SORT", "LiveCellX"):
        for threshold in IOU_THRESHOLDS:
            relevant = [
                row
                for row in case_rows
                if row["condition"] == condition and row["iou_threshold"] == threshold
            ]
            rates = [row["cell_match_rate"] for row in relevant]
            rows.append(
                {
                    "dataset": "process2",
                    "subset": "TD-1 candidates split by CSNet",
                    "condition": condition,
                    "iou_threshold": threshold,
                    "cases": len(relevant),
                    "matched_cells": sum(row["matched_cells"] for row in relevant),
                    "reference_cells": sum(row["reference_cells"] for row in relevant),
                    "cell_match_rate": float(np.mean(rates)),
                }
            )
    return rows


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, candidate_count):
    rate = {
        (row["condition"], row["iou_threshold"]): row["cell_match_rate"]
        for row in rows
    }
    x = np.arange(len(IOU_THRESHOLDS))
    width = 0.34
    conditions = (
        ("SORT", "SORT", "#3F6DB5"),
        ("LiveCellX", "LiveCellX", "#F28E2B"),
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for index, (condition, display_label, color) in enumerate(conditions):
        values = [rate[(condition, threshold)] for threshold in IOU_THRESHOLDS]
        bars = ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=display_label,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

    ax.set_title("A549 cell match rate", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("Cell match rate", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"IoU >= {threshold}" for threshold in IOU_THRESHOLDS])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticklabels([f"{value:.1f}" for value in np.linspace(0.0, 1.0, 6)])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.8)
    ax.legend(frameon=False, ncol=1, loc="upper right")
    fig.text(
        0.5,
        -0.025,
        (
            f"A549 TD-1 under-segmentation cases split by CSNet (n = {candidate_count}); "
            "adjacent-frame cells serve as temporal references."
        ),
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REBUTTAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig3i_process2_csn_cell_match_rate.{extension}", dpi=300, bbox_inches="tight")
        fig.savefig(REBUTTAL_FIG_DIR / f"fig3i_cell_match_iou_thresholds.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    input_paths = sorted_masks(INPUT_MASK_DIR, "*_cp_masks.png")
    corrected_paths = sorted_masks(CORRECTED_MASK_DIR, "frame_*.png")
    if len(input_paths) != len(corrected_paths):
        raise ValueError(f"Mask sequence length mismatch: {len(input_paths)} vs {len(corrected_paths)}")

    print(f"Loading {len(input_paths)} process2 mask pairs...")
    print("Selecting process2 TD-1 temporal under-segmentation cases...")
    candidates = find_td1_candidates(input_paths)
    csn_split_candidates = select_csn_split_candidates(candidates, input_paths, corrected_paths)
    print(f"Found {len(candidates)} TD-1 candidate target objects")
    print(f"Found {len(csn_split_candidates)} candidates split by CSNet")
    if not csn_split_candidates:
        raise RuntimeError("No CSNet-split process2 TD-1 candidates found")

    case_rows = evaluate_candidates(csn_split_candidates, input_paths, corrected_paths)
    rows = aggregate_rows(case_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(CASE_CSV_PATH, case_rows)
    save_csv(RATE_CSV_PATH, rows)
    plot(rows, len(csn_split_candidates))

    for row in rows:
        print(
            f"{row['condition']}, IoU >= {row['iou_threshold']}: "
            f"{row['cell_match_rate'] * 100:.2f}% mean per-case rate "
            f"({row['matched_cells']}/{row['reference_cells']} pooled matched reference cells; "
            f"{row['cases']} cases)"
        )
    print(f"Saved {RATE_CSV_PATH}")
    print(f"Saved {FIG_DIR / 'fig3i_process2_csn_cell_match_rate.pdf'}")
    print(f"Updated {REBUTTAL_FIG_DIR / 'fig3i_cell_match_iou_thresholds.pdf'}")


if __name__ == "__main__":
    main()
