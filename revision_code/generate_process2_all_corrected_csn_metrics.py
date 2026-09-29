#!/usr/bin/env python
"""
Generate process2 CSNet metrics over all correction-relevant case types.

This is different from the "all cells" figure, which is dominated by already
correct cells.  Here the denominator is correction-relevant cases detected from
the before/after mask relationship:

* under-segmentation: one before-CSNet mask maps to multiple adjacent-frame
  reference cells and CSNet split it into multiple components;
* missing/absent cell: an adjacent-frame reference cell has no before-CSNet
  target mask but has an after-CSNet component;
* over-segmentation: one adjacent-frame reference cell maps to multiple
  before-CSNet target masks and one after-CSNet component.

Each case is evaluated using the paper definitions for cell match rate and
correction success rate.
"""

import csv
import os
from collections import defaultdict
from functools import lru_cache

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-livecellx")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from generate_process2_csn_cell_match_rate import (
    CORRECTED_MASK_DIR,
    FIG_DIR,
    INPUT_MASK_DIR,
    IOU_THRESHOLDS,
    IOMIN_CANDIDATE_THRESHOLD,
    OUTPUT_DIR,
    load_mask,
    object_areas,
    overlap_tuples,
    sorted_masks,
    substantial_references,
)


SUBSET_LABEL = "All CSNet-corrected cases"
CELL_RATE_CSV_PATH = OUTPUT_DIR / "process2_all_corrected_csn_cell_match_rate.csv"
CELL_CASE_CSV_PATH = OUTPUT_DIR / "process2_all_corrected_csn_cell_match_case_summary.csv"
SUCCESS_RATE_CSV_PATH = OUTPUT_DIR / "process2_all_corrected_csn_correction_success_rate.csv"
SUCCESS_CASE_CSV_PATH = OUTPUT_DIR / "process2_all_corrected_csn_correction_success_case_summary.csv"
CASE_TYPE_CSV_PATH = OUTPUT_DIR / "process2_all_corrected_csn_case_type_counts.csv"


def reverse_mapping(pred_to_refs):
    ref_to_preds = defaultdict(list)
    for prediction_label, reference_labels in pred_to_refs.items():
        for reference_label in reference_labels:
            ref_to_preds[reference_label].append(prediction_label)
    return {key: sorted(values) for key, values in ref_to_preds.items()}


def corrected_component_labels(before_mask, after_mask, before_label):
    """Return after-CSNet labels derived from one before-CSNet object."""
    target_binary = np.where(before_mask == before_label, 1, 0).astype(np.uint32)
    after_to_original, _ = substantial_references(target_binary, after_mask)
    return sorted(
        after_label
        for after_label, original_labels in after_to_original.items()
        if 1 in original_labels
    )


def choose_candidate(candidates, key, candidate):
    previous = candidates.get(key)
    if previous is None or candidate["selection_score"] > previous["selection_score"]:
        candidates[key] = candidate


def find_all_corrected_cases(before_paths, after_paths):
    cases = {}
    for target_index, target_path in enumerate(before_paths):
        before_mask = load_mask(target_path)
        after_mask = load_mask(after_paths[target_index])
        adjacent_indices = []
        if target_index > 0:
            adjacent_indices.append(target_index - 1)
        if target_index + 1 < len(before_paths):
            adjacent_indices.append(target_index + 1)

        for reference_index in adjacent_indices:
            reference_mask = load_mask(before_paths[reference_index])
            before_pred_to_refs, before_scores = substantial_references(
                reference_mask,
                before_mask,
                threshold=IOMIN_CANDIDATE_THRESHOLD,
            )
            after_pred_to_refs, after_scores = substantial_references(
                reference_mask,
                after_mask,
                threshold=IOMIN_CANDIDATE_THRESHOLD,
            )
            before_ref_to_preds = reverse_mapping(before_pred_to_refs)
            after_ref_to_preds = reverse_mapping(after_pred_to_refs)

            # Under-segmentation: one target mask represents multiple reference cells.
            for before_label, reference_labels in before_pred_to_refs.items():
                if len(reference_labels) < 2:
                    continue
                after_labels = corrected_component_labels(before_mask, after_mask, before_label)
                if len(after_labels) < 2:
                    continue
                candidate = {
                    "case_type": "under-seg",
                    "target_index": target_index,
                    "reference_index": reference_index,
                    "reference_labels": sorted(reference_labels),
                    "before_prediction_labels": [before_label],
                    "after_prediction_labels": after_labels,
                    "target_label": before_label,
                    "selection_score": float(sum(before_scores.get(before_label, [])) + len(after_labels)),
                }
                choose_candidate(cases, ("under-seg", target_index, before_label), candidate)

            # Missing/absent cell: no before target, but an after target recovers it.
            for reference_label, after_labels in after_ref_to_preds.items():
                if reference_label in before_ref_to_preds:
                    continue
                for after_label in after_labels:
                    candidate = {
                        "case_type": "missing",
                        "target_index": target_index,
                        "reference_index": reference_index,
                        "reference_labels": [reference_label],
                        "before_prediction_labels": [],
                        "after_prediction_labels": [after_label],
                        "target_label": after_label,
                        "selection_score": float(sum(after_scores.get(after_label, []))),
                    }
                    choose_candidate(cases, ("missing", target_index, after_label), candidate)

            # Over-segmentation: multiple before targets represent one reference cell,
            # then CSNet/correction produces one target component for that reference.
            for reference_label, before_labels in before_ref_to_preds.items():
                after_labels = after_ref_to_preds.get(reference_label, [])
                if len(before_labels) < 2 or len(after_labels) != 1:
                    continue
                candidate = {
                    "case_type": "over-seg",
                    "target_index": target_index,
                    "reference_index": reference_index,
                    "reference_labels": [reference_label],
                    "before_prediction_labels": sorted(before_labels),
                    "after_prediction_labels": after_labels,
                    "target_label": after_labels[0],
                    "selection_score": float(len(before_labels) + sum(after_scores.get(after_labels[0], []))),
                }
                choose_candidate(cases, ("over-seg", target_index, tuple(sorted(before_labels))), candidate)

        if (target_index + 1) % 50 == 0:
            print(f"  Scanned {target_index + 1} frames")
    return list(cases.values())


@lru_cache(maxsize=16)
def overlap_context(reference_index, target_index, condition):
    before_paths = overlap_context.before_paths
    after_paths = overlap_context.after_paths
    reference_mask = load_mask(before_paths[reference_index])
    prediction_mask = load_mask(before_paths[target_index]) if condition == "SORT" else load_mask(after_paths[target_index])
    reference_areas = object_areas(reference_mask)
    prediction_areas = object_areas(prediction_mask)
    intersections = {
        (reference_label, prediction_label): intersection
        for reference_label, prediction_label, intersection in overlap_tuples(reference_mask, prediction_mask)
    }
    return reference_areas, prediction_areas, intersections


def restricted_best_iou(reference_index, target_index, condition, reference_labels, prediction_labels):
    reference_areas, prediction_areas, intersections = overlap_context(reference_index, target_index, condition)
    if not prediction_labels:
        return np.zeros(len(reference_labels), dtype=float), 0

    ref_position = {label: index for index, label in enumerate(reference_labels)}
    pred_position = {label: index for index, label in enumerate(prediction_labels)}
    iou_matrix = np.zeros((len(reference_labels), len(prediction_labels)), dtype=float)

    for reference_label in reference_labels:
        for prediction_label in prediction_labels:
            intersection = intersections.get((reference_label, prediction_label), 0)
            if intersection == 0:
                continue
            union = reference_areas[reference_label] + prediction_areas[prediction_label] - intersection
            iou_matrix[ref_position[reference_label], pred_position[prediction_label]] = intersection / union

    row_indices, column_indices = linear_sum_assignment(-iou_matrix)
    best_iou = np.zeros(len(reference_labels), dtype=float)
    for row, column in zip(row_indices, column_indices):
        best_iou[row] = iou_matrix[row, column]
    return best_iou, len(prediction_labels)


def evaluate_cases(cases):
    cell_case_rows = []
    success_case_rows = []
    for case_id, case in enumerate(cases, start=1):
        for condition, prediction_labels in (
            ("SORT", case["before_prediction_labels"]),
            ("LiveCellX", case["after_prediction_labels"]),
        ):
            best_iou, prediction_count = restricted_best_iou(
                case["reference_index"],
                case["target_index"],
                condition,
                case["reference_labels"],
                prediction_labels,
            )
            for threshold in IOU_THRESHOLDS:
                matched = int(np.sum(best_iou >= threshold))
                success = int(
                    prediction_count == len(case["reference_labels"])
                    and matched == len(case["reference_labels"])
                )
                common = {
                    "case_id": case_id,
                    "case_type": case["case_type"],
                    "condition": condition,
                    "iou_threshold": threshold,
                    "target_frame": case["target_index"] + 1,
                    "reference_frame": case["reference_index"] + 1,
                    "target_label": case["target_label"],
                    "reference_cells": len(case["reference_labels"]),
                    "prediction_cells": prediction_count,
                }
                cell_case_rows.append(
                    {
                        **common,
                        "matched_cells": matched,
                        "cell_match_rate": matched / len(case["reference_labels"]),
                    }
                )
                success_case_rows.append(
                    {
                        **common,
                        "success": success,
                        "matched_pairs": matched,
                    }
                )
    return cell_case_rows, success_case_rows


def aggregate_cell_rows(case_rows):
    rows = []
    for condition in ("SORT", "LiveCellX"):
        for threshold in IOU_THRESHOLDS:
            relevant = [
                row
                for row in case_rows
                if row["condition"] == condition and row["iou_threshold"] == threshold
            ]
            rows.append(
                {
                    "dataset": "process2",
                    "subset": SUBSET_LABEL,
                    "condition": condition,
                    "iou_threshold": threshold,
                    "cases": len(relevant),
                    "matched_cells": sum(row["matched_cells"] for row in relevant),
                    "reference_cells": sum(row["reference_cells"] for row in relevant),
                    "cell_match_rate": float(np.mean([row["cell_match_rate"] for row in relevant])),
                }
            )
    return rows


def aggregate_success_rows(case_rows):
    rows = []
    for condition in ("SORT", "LiveCellX"):
        for threshold in IOU_THRESHOLDS:
            relevant = [
                row
                for row in case_rows
                if row["condition"] == condition and row["iou_threshold"] == threshold
            ]
            success_count = sum(row["success"] for row in relevant)
            rows.append(
                {
                    "dataset": "process2",
                    "subset": SUBSET_LABEL,
                    "condition": condition,
                    "iou_threshold": threshold,
                    "cases": len(relevant),
                    "successful_cases": success_count,
                    "correction_success_rate": success_count / len(relevant),
                }
            )
    return rows


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bars(rows, metric_key, title, ylabel, xlabel, output_stem, footnote):
    rate = {
        (row["condition"], row["iou_threshold"]): row[metric_key]
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

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
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
    fig.text(0.5, -0.025, footnote, ha="center", fontsize=8.2, color="#555555")
    fig.tight_layout(rect=[0, 0.10, 1, 1])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{output_stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    before_paths = sorted_masks(INPUT_MASK_DIR, "*_cp_masks.png")
    after_paths = sorted_masks(CORRECTED_MASK_DIR, "frame_*.png")
    if len(before_paths) != len(after_paths):
        raise ValueError(f"Mask sequence length mismatch: {len(before_paths)} vs {len(after_paths)}")

    overlap_context.before_paths = tuple(before_paths)
    overlap_context.after_paths = tuple(after_paths)

    print(f"Loading {len(before_paths)} process2 mask pairs...")
    print("Selecting all correction-relevant process2 cases...")
    cases = find_all_corrected_cases(before_paths, after_paths)
    cases = sorted(
        cases,
        key=lambda case: (case["case_type"], case["target_index"], case["reference_index"], case["target_label"]),
    )
    case_type_counts = []
    for case_type in sorted({case["case_type"] for case in cases}):
        count = sum(case["case_type"] == case_type for case in cases)
        case_type_counts.append({"case_type": case_type, "cases": count})
        print(f"  {case_type}: {count}")
    print(f"Found {len(cases)} total correction-relevant cases")

    print("Evaluating cell-match and correction-success metrics...")
    cell_case_rows, success_case_rows = evaluate_cases(cases)
    cell_rows = aggregate_cell_rows(cell_case_rows)
    success_rows = aggregate_success_rows(success_case_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(CASE_TYPE_CSV_PATH, case_type_counts)
    save_csv(CELL_CASE_CSV_PATH, cell_case_rows)
    save_csv(CELL_RATE_CSV_PATH, cell_rows)
    save_csv(SUCCESS_CASE_CSV_PATH, success_case_rows)
    save_csv(SUCCESS_RATE_CSV_PATH, success_rows)

    type_text = ", ".join(f"{row['case_type']}={row['cases']}" for row in case_type_counts)
    footnote = (
        f"Process2 CSNet correction-relevant cases (n = {len(cases)}; {type_text}); "
        "adjacent-frame cells serve as temporal references."
    )
    plot_grouped_bars(
        cell_rows,
        metric_key="cell_match_rate",
        title="Process2 cell match rate",
        ylabel="Cell match rate",
        xlabel="",
        output_stem="fig3i_process2_all_corrected_csn_cell_match_rate",
        footnote=footnote,
    )
    plot_grouped_bars(
        success_rows,
        metric_key="correction_success_rate",
        title="Process2 correction success rate",
        ylabel="Correction success rate",
        xlabel="",
        output_stem="fig3i_process2_all_corrected_csn_correction_success_rate",
        footnote=footnote + " Success requires perfect one-to-one matching.",
    )

    for rows, key, name in (
        (cell_rows, "cell_match_rate", "Cell match"),
        (success_rows, "correction_success_rate", "Correction success"),
    ):
        print(name)
        for row in rows:
            count_text = (
                f"{row['matched_cells']}/{row['reference_cells']} matched cells"
                if key == "cell_match_rate"
                else f"{row['successful_cases']}/{row['cases']} successful cases"
            )
            print(
                f"  {row['condition']}, IoU >= {row['iou_threshold']}: "
                f"{row[key] * 100:.2f}% ({count_text})"
            )

    print(f"Saved {FIG_DIR / 'fig3i_process2_all_corrected_csn_cell_match_rate.pdf'}")
    print(f"Saved {FIG_DIR / 'fig3i_process2_all_corrected_csn_correction_success_rate.pdf'}")


if __name__ == "__main__":
    main()
