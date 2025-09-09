#!/usr/bin/env python
"""
Multi-round Correction with Tracking Benchmark
Based on CXA_2D_process2-step3_tracking_correction.py

This script combines tracking benchmark (multiple max_age parameters) with
multi-round undersegmentation correction. It preserves the original multi-round
correction loop to avoid object reference bugs.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from scipy.optimize import linear_sum_assignment

from livecellx.core import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection
from livecellx.core.io_utils import save_png
from livecellx.core.single_cell import get_time2scs
from livecellx.core.sc_filters import filter_boundary_traj
from livecellx.model_zoo.segmentation.eval_csn import compute_watershed
from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux
from livecellx.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset
from livecellx.preprocess.utils import normalize_img_to_uint8, overlay
from livecellx.segment.ou_simulator import find_label_mask_contours
from livecellx.track.sort_tracker_utils import track_SORT_bbox_from_scs
from livecellx.model_zoo.segmentation.csn_sc_utils import correct_sc
from livecellx.model_zoo.segmentation.custom_transforms import CustomTransformEdtV9
from torchvision import transforms


# Match hyperparameters from CXA_2D_process2_step3_apply_all.py
PADDING_PIXELS = 20
OUT_THRESHOLD = 1


def get_parser():
    parser = argparse.ArgumentParser(description="Multi-round correction with tracking benchmark")

    # Required arguments
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for all experiment outputs")
    parser.add_argument(
        "--lcx_data_dir",
        type=str,
        required=True,
        help="Directory containing input data (e.g., notebook_results/CXA_process2_7_19)",
    )
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to CSN model checkpoint")

    # Tracking parameters
    parser.add_argument(
        "--max_ages",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 40, 80, 160],
        help="List of max_age values for tracking benchmark",
    )
    parser.add_argument("--min_hits", type=int, default=1, help="Min hits for SORT tracker")

    # Multi-round correction parameters
    parser.add_argument("--match_threshold", type=float, default=0.5, help="Match threshold for underseg detection")
    parser.add_argument("--match_search_interval", type=int, default=3, help="Search interval for matching")
    parser.add_argument("--max_round", type=int, default=100, help="Maximum rounds of correction per tracking result")
    parser.add_argument("--area_threshold", type=float, default=1000, help="Area threshold for filtering small regions")
    parser.add_argument("--dist_to_boundary", type=int, default=50, help="Distance to boundary for filtering")

    # CSN model parameters
    parser.add_argument("--h_threshold", type=float, default=0.3, help="H threshold for watershed")
    parser.add_argument("--out_threshold", type=float, default=1.0, help="Output threshold for watershed")
    parser.add_argument("--padding", type=int, default=20, help="Padding for cell crops")

    # Input file names (relative to lcx_data_dir)
    parser.add_argument(
        "--single_cells_json", type=str, default="single_cells_zero_based_time.json", help="Single cells JSON file name"
    )
    parser.add_argument(
        "--annotation_file", type=str, default="multimap_annotation-Ke-SJ-10-15.csv", help="Annotation CSV file name"
    )
    parser.add_argument(
        "--precomputed_sctc",
        type=str,
        default="sctc_filled_SORT_bbox_max_age_3_min_hits_1.json",
        help="Pre-computed SCTC file name (if available)",
    )

    # Options
    parser.add_argument(
        "--save_masks", action="store_true", help="Save individual masks (not recommended, uses lots of space)"
    )
    parser.add_argument("--DEBUG", action="store_true", help="Debug mode with limited cells")
    parser.add_argument(
        "--enable_viz",
        action="store_true",
        default=False,
        help="Enable visualization plots (disabled by default for large datasets)",
    )
    parser.add_argument(
        "--run_cxa_experiment",
        action="store_true",
        default=True,
        help="Run CXA experiment from LivecellX paper (default: True)",
    )
    parser.add_argument(
        "--force_retrack", action="store_true", default=False, help="Force re-run tracking even if cached results exist"
    )
    parser.add_argument(
        "--minimal_output",
        action="store_true",
        default=True,
        help="Minimize disk usage by not creating per-round directories or duplicate files",
    )
    parser.add_argument(
        "--skip_tracking", action="store_true", default=False, help="Skip tracking phase and use pre-existing SCTC file"
    )
    parser.add_argument(
        "--input_sctc",
        type=str,
        default=None,
        help="Path to pre-existing SCTC JSON file to use (requires --skip_tracking)",
    )

    return parser


###############################################################################
# Helper functions from original script (keeping them intact)
###############################################################################


def get_sc_before_time(sct, time: float):
    """Get single cell before given time in trajectory"""
    cur_time = time
    times: Union[List[int], List[float]] = list(sct.timeframe_to_single_cell.keys())
    prev_time = None
    min_diff = None
    for time in times:
        if time < cur_time:
            if min_diff is None or cur_time - time < min_diff:
                min_diff = cur_time - time
                prev_time = time
    if prev_time is None:
        return None
    return sct.timeframe_to_single_cell[prev_time]


def get_sc_after_time(sct, time: float):
    """Get single cell after given time in trajectory"""
    cur_time = time
    times: Union[List[int], List[float]] = list(sct.timeframe_to_single_cell.keys())
    next_time = None
    min_diff = None
    for time in times:
        if time > cur_time:
            if min_diff is None or time - cur_time < min_diff:
                min_diff = time - cur_time
                next_time = time
    if next_time is None:
        return None
    return sct.timeframe_to_single_cell[next_time]


def show_tracks(sctc: SingleCellTrajectoryCollection, denoted_scs, figsize=(5, 40), y_interval=10):
    """Visualize trajectories with marked cells"""
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    track_y = 0
    denoted_scs = set(denoted_scs)
    scts = sctc.get_all_trajectories()
    sorted_scts = sorted(scts, key=lambda x: (x.get_time_span()[0], len(x)), reverse=False)
    for sct in sorted_scts:
        _scs = list(sct.timeframe_to_single_cell.values())
        times = [sc.timeframe for sc in _scs]
        ax.plot(
            times,
            [track_y] * len(times),
            marker="o",
            linestyle="-",
            color="blue",
            markersize=2,
            linewidth=1,
        )
        _denoted_scs = [sc for sc in _scs if sc in denoted_scs]
        denoted_times = [sc.timeframe for sc in _denoted_scs]
        ax.scatter(denoted_times, [track_y] * len(denoted_times), marker="o", color="red", s=10)
        track_y += y_interval
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Track ID")
    ax.set_ylim(0, track_y)
    return fig, ax


def show_tracks_missing(sctc: SingleCellTrajectoryCollection, denoted_pairs, figsize=(5, 40), y_interval=10):
    """
    Visualizes the missing tracks in a SingleCellTrajectoryCollection.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    track_y = 0
    denoted_pairs_prev = [pair[0] for pair in denoted_pairs]
    denoted_to2prev = {pair[1]: pair[0] for pair in denoted_pairs}
    denoted_prev2times = {}
    for prev_sc, to_sc in denoted_pairs:
        if prev_sc not in denoted_prev2times:
            denoted_prev2times[prev_sc] = []
        denoted_prev2times[prev_sc].append(to_sc.timeframe)

    denoted_prev_scs = set(denoted_pairs_prev)
    scts = sctc.get_all_trajectories()
    sorted_scts = sorted(scts, key=lambda x: (x.get_time_span()[0], len(x)), reverse=False)
    for sct in sorted_scts:
        _scs = list(sct.timeframe_to_single_cell.values())
        times = [sc.timeframe for sc in _scs]
        ax.plot(
            times,
            [track_y] * len(times),
            marker="o",
            linestyle="-",
            color="blue",
            markersize=2,
            linewidth=1,
        )
        _denoted_scs = [sc for sc in _scs if sc in denoted_prev_scs]
        denoted_times_underseg = []
        for _sc in _denoted_scs:
            denoted_times_underseg += denoted_prev2times[_sc]
        denoted_times_prev = list(set([sc.timeframe for sc in _denoted_scs]))
        ax.scatter(
            denoted_times_underseg,
            [track_y] * len(denoted_times_underseg),
            marker="x",
            color="red",
            s=10,
        )
        ax.scatter(
            denoted_times_prev,
            [track_y] * len(denoted_times_prev),
            marker="o",
            color="green",
            s=20,
        )
        track_y += y_interval
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Track ID")
    ax.set_ylim(0, track_y)
    return fig, ax


###########################################################################
# CSN correction module (from original script)
###########################################################################


def get_sc_mask_path(sc: SingleCellStatic, out_dir: Path) -> Path:
    """Get the output path for a single cell mask"""
    return out_dir / "watershed_masks" / f"mask_{sc.timeframe}_{sc.id}.npy"


def get_sc_watershed_path(sc: SingleCellStatic, out_dir: Path) -> Path:
    """Get the output path for a single cell watershed mask"""
    return out_dir / "watershed_masks" / f"watershed_{sc.timeframe}_{sc.id}.npy"


def is_sc_watershed_mask_exists(sc: SingleCellStatic) -> bool:
    """Check if watershed mask exists for a single cell"""
    # Original script expects mask in specific location
    mask_path = Path(
        f"notebook_results/CXA_process2_7_19/correction_watershed_masks/watershed_{sc.timeframe}_{sc.id}.png"
    )
    return mask_path.exists()


def read_sc_watershed_mask(sc: SingleCellStatic) -> np.ndarray:
    """Read watershed mask for a single cell"""
    # Original script expects mask in specific location
    mask_path = Path(
        f"notebook_results/CXA_process2_7_19/correction_watershed_masks/watershed_{sc.timeframe}_{sc.id}.png"
    )
    if not mask_path.exists():
        return None
    return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


def compute_sc_watershed_mask(
    sc, model, input_transforms, padding=PADDING_PIXELS, h_threshold=1, out_threshold=OUT_THRESHOLD
):
    """Compute watershed mask for a single cell using CSN model - matches original CXA script approach"""
    # Use correct_sc function but with min_area=4000 to match the original script
    # Note: The original script uses min_area=4000, not the library default of 100
    res_dict = correct_sc(
        sc, model, padding, input_transforms, gpu=True, return_outputs=True, h_threshold=h_threshold, min_area=4000
    )

    # Extract watershed mask from results
    watershed_mask = res_dict["watershed_mask"]

    return watershed_mask


def gen_missing_case_masks(
    in_sc,
    missing_sc,
    model,
    input_transforms,
    out_dir,
    padding=PADDING_PIXELS,
    h_threshold=0.5,
    out_threshold=OUT_THRESHOLD,
):
    """Generate masks for missing case - saves to disk only if save_masks is True"""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "correction_figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Always compute masks - Note: original script doesn't pass h_threshold to correct_sc, so it uses default h_threshold=1
    _in_watershed = compute_sc_watershed_mask(in_sc, model, input_transforms, padding)
    _missing_watershed = compute_sc_watershed_mask(missing_sc, model, input_transforms, padding)

    # Only save if explicitly requested (controlled by caller)
    if hasattr(gen_missing_case_masks, "save_masks") and gen_missing_case_masks.save_masks:
        mask_dir = out_dir / "watershed_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        np.save(get_sc_watershed_path(in_sc, out_dir), _in_watershed)
        np.save(get_sc_watershed_path(missing_sc, out_dir), _missing_watershed)

        # Save example visualizations
        save_png(_in_watershed, fig_dir / f"watershed_in_sc_{in_sc.timeframe}_{in_sc.id}.png")
        save_png(_missing_watershed, fig_dir / f"watershed_missing_sc_{missing_sc.timeframe}_{missing_sc.id}.png")

    return _in_watershed, _missing_watershed


def overwrite_sct(target_sct: SingleCellTrajectory, src_sct: SingleCellTrajectory, timeframe, inplace=False):
    if not inplace:
        target_sct = target_sct.copy()
    if timeframe in target_sct.timeframe_set:
        target_sct.pop_sc_by_time(timeframe)
    if timeframe in src_sct.timeframe_set:
        target_sct.add_single_cell(src_sct.get_single_cell(timeframe))
    else:
        print("Warning: Timeframe not found in src_sct")
    return target_sct


def select_underseg_cells_by_missing(sctc, search_interval=2, threshold=0.5):
    """Select undersegmented cells by missing logic"""
    prev_or_after_matched_scs = set()
    underseg_candidate_pairs = []
    trajectories = sctc.get_all_trajectories()
    all_scs = sctc.get_all_scs()
    time2scs = get_time2scs(all_scs)

    for sct in tqdm.tqdm(trajectories, desc="Selecting under-seg candidates by matching other cells"):
        timespan = sct.get_time_span()
        head = timespan[0]
        end = timespan[1]

        # Prev logics
        for time in range(head, end + 1):
            if time not in sct.timeframe_to_single_cell:
                prev_sc = get_sc_before_time(sct, time)
                if prev_sc is None:
                    continue
                if int(prev_sc.timeframe) + search_interval < time:
                    continue
                if time not in time2scs:
                    continue

                prev_or_after_matched_scs.add(prev_sc)
                scs_at_time = time2scs[time]
                for cur_sc in scs_at_time:
                    metric = cur_sc.compute_iomin(prev_sc)
                    if metric > threshold:
                        underseg_candidate_pairs.append((prev_sc, cur_sc))

        # After logic
        for time in range(head, end + 1):
            if time not in sct.timeframe_to_single_cell:
                next_sc = get_sc_after_time(sct, time)
                if next_sc is None:
                    continue
                if int(next_sc.timeframe) - search_interval >= time:
                    continue
                if time not in time2scs:
                    continue
                prev_or_after_matched_scs.add(next_sc)
                scs_at_time = time2scs[time]
                for cur_sc in scs_at_time:
                    metric = cur_sc.compute_iomin(next_sc)
                    if metric > threshold:
                        underseg_candidate_pairs.append((next_sc, cur_sc))

    print(f"Number of underseg_candidate_pairs by missing logics: {len(underseg_candidate_pairs)}")
    print(f"Number of prev_or_after_matched_scs by missing logics: {len(prev_or_after_matched_scs)}")
    return {"underseg_candidate_pairs": underseg_candidate_pairs, "prev_or_after_scs": prev_or_after_matched_scs}


def select_underseg_cells_by_end(sctc, search_interval=2, threshold=0.5):
    """Select undersegmented cells by ending logic"""
    underseg_candidate_pairs = []
    end_matched_scs = set()
    trajectories = sctc.get_all_trajectories()
    all_scs = sctc.get_all_scs()
    time2scs = get_time2scs(all_scs)

    for sct in tqdm.tqdm(trajectories, desc="Selecting under-seg candidates by matching end cells"):
        timespan = sct.get_time_span()
        end = timespan[1]

        for time_offset in range(1, search_interval + 1):
            time = end + time_offset
            if time not in time2scs:
                continue

            end_sc = sct.get_single_cell(end)
            end_matched_scs.add(end_sc)
            scs_at_time = time2scs[time]

            for cur_sc in scs_at_time:
                metric = cur_sc.compute_iomin(end_sc)
                if metric > threshold:
                    underseg_candidate_pairs.append((end_sc, cur_sc))

    print(f"Number of underseg_candidate_pairs by ending logics: {len(underseg_candidate_pairs)}")
    print(f"Number of end_matched_scs by ending logics: {len(end_matched_scs)}")
    return {"underseg_candidate_pairs": underseg_candidate_pairs, "end_cells": end_matched_scs}


def report_underseg_candidates(underseg_candidates, predicted_scs, all_scs):
    underseg_candidates_not_in_gt = [sc for sc in underseg_candidates if sc not in predicted_scs]
    underseg_candidates_in_gt = [sc for sc in underseg_candidates if sc in predicted_scs]

    print("# of total cells:", len(all_scs))
    print("# of total GT multmaps cells:", len(predicted_scs))
    print(
        "Coverage of GT multimap cells:",
        len(underseg_candidates_in_gt) / len(predicted_scs) if len(predicted_scs) > 0 else 0,
    )
    print(
        "# of total U-S candidates by logics:",
        len(underseg_candidates),
    )
    print("# of underseg candidates in GT:", len(underseg_candidates_in_gt))
    print("# of underseg candidates not in GT:", len(underseg_candidates_not_in_gt))


def cost_iou(sc1, sc2):
    return -sc1.compute_iou(sc2)


def match_scs_by_lap(scs_1, scs_2, cost_function):
    """Match two lists of single cells using the Linear Assignment Problem (LAP)."""
    cost_matrix = np.zeros((len(scs_1), len(scs_2)))
    for i, sc1 in enumerate(scs_1):
        for j, sc2 in enumerate(scs_2):
            cost_matrix[i, j] = cost_function(sc1, sc2)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = [(i, j) for i, j in zip(row_ind, col_ind)]
    sc1_to_sc2 = {scs_1[i]: scs_2[j] for i, j in matches}
    return sc1_to_sc2


def contours_to_scs(contours, ref_sc: SingleCellStatic, padding=None, min_area=None):
    """Convert contours to SingleCellStatic objects"""
    new_scs = []
    sc_bbox = ref_sc.get_bbox(padding=padding)
    for contour in contours:
        if min_area is not None and cv2.contourArea(contour) < min_area:
            continue
        _contour = np.array([(sc_bbox[0] + x, sc_bbox[1] + y) for x, y in contour])
        new_sc = SingleCellStatic(
            ref_sc.timeframe,
            contour=_contour,
            img_dataset=ref_sc.img_dataset,
        )
        new_scs.append(new_sc)
    return new_scs


def fix_missing_trajectory(
    in_sc,
    missing_sc,
    padding=PADDING_PIXELS,
    area_threshold=1000,
    inplace=False,
    model=None,
    input_transforms=None,
    h_threshold=0.5,
    out_threshold=OUT_THRESHOLD,
    verbose=False,
):
    """Fix missing trajectory - with on-the-fly mask generation"""
    sct_in_orig = in_sc.tmp["sct"]
    sct_missing_orig = missing_sc.tmp["sct"]

    if not inplace:
        sct_in = sct_in_orig.copy()
        sct_missing = sct_missing_orig.copy()

    # Check if masks exist, if not generate them on-the-fly
    watershed_mask_in = None
    watershed_mask_missing = None

    if is_sc_watershed_mask_exists(in_sc):
        watershed_mask_in = read_sc_watershed_mask(in_sc)
    else:
        # Generate mask on-the-fly
        if model is not None and input_transforms is not None:
            if verbose:
                print(f"Generating watershed mask for {in_sc.id} on-the-fly...")
            watershed_mask_in = compute_sc_watershed_mask(
                in_sc, model, input_transforms, padding, h_threshold, out_threshold
            )
        else:
            print(f"Missing mask for {in_sc.id} and no model provided to generate")
            assert False, "No model provided to generate missing mask"

    if is_sc_watershed_mask_exists(missing_sc):
        watershed_mask_missing = read_sc_watershed_mask(missing_sc)
    else:
        # Generate mask on-the-fly
        if model is not None and input_transforms is not None:
            if verbose:
                print(f"Generating watershed mask for {missing_sc.id} on-the-fly...")
            watershed_mask_missing = compute_sc_watershed_mask(
                missing_sc, model, input_transforms, padding, h_threshold, out_threshold
            )
        else:
            print(f"Missing mask for {missing_sc.id} and no model provided to generate")
            assert False, "No model provided to generate missing mask"

    assert watershed_mask_in is not None, "Watershed mask for in_sc is None"
    assert watershed_mask_missing is not None, "Watershed mask for missing_sc is None"

    in_contours = find_label_mask_contours(watershed_mask_in)
    missing_contours = find_label_mask_contours(watershed_mask_missing)

    # Filter out small regions
    in_contours = [_contour for _contour in in_contours if cv2.contourArea(_contour) > area_threshold]
    missing_contours = [_contour for _contour in missing_contours if cv2.contourArea(_contour) > area_threshold]

    case_type = None
    res_dict = {}
    if len(in_contours) == 1 and len(missing_contours) > 1:
        case_type = "missing-US"
        new_scs = contours_to_scs(missing_contours, missing_sc, padding=padding)
        res_dict["new_scs"] = new_scs
        if missing_sc.timeframe not in sct_missing.timeframe_to_single_cell:
            print(sct_missing.timeframe_to_single_cell.keys())
            print("missing time:", missing_sc.timeframe)
            print("is missing sc in sct_missing?", missing_sc in sct_missing.get_all_scs())

        # Now we have a list of new_scs, we need to match them to sct_missing and sct_in
        if missing_sc.timeframe in sct_missing.timeframe_to_single_cell:
            sct_missing.timeframe_to_single_cell.pop(missing_sc.timeframe)

        # Tricky here: we do not want to use missing_sc because it is potentially an underseg cell mask
        # Instead, we use the previous sc to match the new_scs, assuming prev/after sc is likely correct
        temporal_neighbor_missing_sc = get_sc_before_time(sct_missing, missing_sc.timeframe)
        if temporal_neighbor_missing_sc is None:
            temporal_neighbor_missing_sc = get_sc_after_time(sct_missing, missing_sc.timeframe)

        is_lacking_neighbor = False
        if temporal_neighbor_missing_sc is None:
            is_lacking_neighbor = True
            temporal_neighbor_missing_sc = missing_sc
            lsa_mapping = match_scs_by_lap(new_scs, [in_sc, missing_sc], cost_iou)
        else:
            lsa_mapping = match_scs_by_lap(new_scs, [in_sc, temporal_neighbor_missing_sc], cost_iou)
        # Add new_scs to trajectories based on mapping
        mapped_orig_scs = set()
        in_matched = False
        other_matched = False
        for new_sc in new_scs:
            if new_sc not in lsa_mapping:
                continue
            mapped_sc = lsa_mapping[new_sc]

            mapped_orig_scs.add(mapped_sc)
            if mapped_sc == in_sc:
                new_sc.timeframe = missing_sc.timeframe
                sct_in.add_single_cell(new_sc)
                in_matched = True
            elif mapped_sc == temporal_neighbor_missing_sc:
                new_sc.timeframe = missing_sc.timeframe
                sct_missing.add_single_cell(new_sc)
                other_matched = True
            else:
                assert False, "Unexpected mapped sc"
        assert in_matched, "in_sc not matched"
        assert other_matched, "other sc not matched"
        # Check if all the underseg pair scs are mapped
        assert (
            len(mapped_orig_scs) == 2
        ), f"Not all the underseg pair scs are mapped:{mapped_orig_scs}, \n {lsa_mapping}"
        return {
            "case_type": case_type,
            "new_scs": new_scs,
            "lsa_map": lsa_mapping,
            "updated_scts": [sct_in, sct_missing],
            "state": "fixed",
            "is_lacking_neighbor": is_lacking_neighbor,
            "sc_pair": (in_sc, missing_sc),
        }
    elif len(missing_contours) and len(in_contours) > 1:
        case_type = "P/A-US"
        return {"case_type": case_type, "state": "skipped"}
    elif len(missing_contours) == 1 and len(in_contours) == 1:
        case_type = "Both-correct"
        return {"case_type": case_type, "state": "skipped"}
    elif len(missing_contours) > 1 and len(in_contours) > 1:
        case_type = "Both-US"
        return {"case_type": case_type, "state": "skipped"}
    else:
        case_type = "Others"
        return {"case_type": case_type, "state": "skipped"}


def get_missing_rate(sct: SingleCellTrajectory):
    timespan = sct.get_time_span_length()
    tracked_cells = len(sct.get_all_scs())
    return 1 - tracked_cells / timespan


def compute_sctc_missing_rate(sctc: SingleCellTrajectoryCollection):
    missing_rates = {}
    for _, sct in sctc:
        missing_rates[sct.track_id] = get_missing_rate(sct)
    return missing_rates


def save_trajectory_lengths_csv(sctc: SingleCellTrajectoryCollection, output_path: Path, round_num: int):
    """Save trajectory lengths to CSV file for a given round"""
    trajectory_data = []

    for tid, sct in sctc:
        times = sct.times  # Use the times attribute directly
        if len(times) == 0:
            continue

        trajectory_data.append(
            {
                "trajectory_id": tid,
                "round": round_num,
                "length": len(times),
                "start_time": times[0] if times else None,
                "end_time": times[-1] if times else None,
                "time_span": times[-1] - times[0] + 1 if times else 0,
                "missing_frames": (times[-1] - times[0] + 1 - len(times)) if times else 0,
                "vacancy_rate": ((times[-1] - times[0] + 1 - len(times)) / (times[-1] - times[0] + 1))
                if times and (times[-1] - times[0] + 1) > 0
                else 0,
            }
        )

    df = pd.DataFrame(trajectory_data)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved trajectory lengths for round {round_num} to {output_path}")
    return df


###############################################################################
# Main tracking benchmark and multi-round correction
###############################################################################


def run_tracking_benchmark(all_scs, max_ages, min_hits, output_dir, force_retrack=False, args=None):
    """Run tracking with different max_age parameters with caching support"""
    tracking_files = {}  # Just store file paths, not SCTCs
    tracking_summary_all = []

    # Set up caching directory
    cached_tracking_dir = Path("./notebook_results/CXA_process2_7_19/correction_tracking_sctc_inputs")
    cached_tracking_dir.mkdir(parents=True, exist_ok=True)

    for max_age in max_ages:
        # Check for cached tracking result
        cache_sctc_file = cached_tracking_dir / f"tracking_max_age_{max_age}_sctc.json"
        cache_stats_file = cached_tracking_dir / f"tracking_max_age_{max_age}_stats.json"

        if cache_sctc_file.exists() and cache_stats_file.exists() and not force_retrack:
            print(f"\nFound cached tracking result for max_age={max_age}")
            # Just store the file path, don't load the SCTC yet
            tracking_files[max_age] = {"sctc_file": cache_sctc_file, "stats_file": cache_stats_file, "cached": True}

            # Load just the stats to display info
            try:
                with open(cache_stats_file, "r") as f:
                    stats = json.load(f)

                print(f"  Trajectories: {stats['num_trajectories']}")
                print(f"  Cells: {stats['num_cells']}")
                print(f"  Average missing rate: {stats['avg_missing_rate']:.4f}")

                # Store summary
                tracking_summary = {
                    "max_age": max_age,
                    "num_trajectories": stats["num_trajectories"],
                    "num_cells": stats["num_cells"],
                    "avg_missing_rate": stats["avg_missing_rate"],
                    "completion_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                tracking_summary_all.append(tracking_summary)
                continue
            except Exception as e:
                print(f"  Warning: Failed to load stats ({e}). Re-tracking...")

        print(f"\nTracking with max_age={max_age}...")

        # Track cells - need to provide raw_imgs dataset
        # Get the image dataset from single cells
        if all_scs:
            raw_imgs = all_scs[0].img_dataset
        else:
            raise ValueError("No single cells provided for tracking")

        sctc_tracked = track_SORT_bbox_from_scs(
            all_scs,
            raw_imgs=raw_imgs,
            max_age=max_age,
            min_hits=min_hits,
            sc_inplace=True,
        )

        # Calculate statistics
        missing_rates = compute_sctc_missing_rate(sctc_tracked)
        avg_missing_rate = np.mean(list(missing_rates.values()))

        # Store tracking info temporarily
        num_trajectories = len(sctc_tracked)
        num_cells = len(sctc_tracked.get_all_scs())

        print(f"  Trajectories: {num_trajectories}")
        print(f"  Cells: {num_cells}")
        print(f"  Average missing rate: {avg_missing_rate:.4f}")

        if not args.minimal_output:
            # Save tracking result to output directory
            tracking_out_dir = output_dir / f"max_age_{max_age}"
            tracking_out_dir.mkdir(exist_ok=True, parents=True)
            sctc_tracked.write_json(tracking_out_dir / "sctc_tracked.json")

        # Save individual tracking summary
        tracking_summary = {
            "max_age": max_age,
            "num_trajectories": num_trajectories,
            "num_cells": num_cells,
            "avg_missing_rate": avg_missing_rate,
            "completion_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        tracking_summary_all.append(tracking_summary)
        if not args.minimal_output:
            # Save individual tracking summary
            with open(tracking_out_dir / "tracking_summary.json", "w") as f:
                json.dump(tracking_summary, f, indent=4)

        # Save to cache
        print(f"  Saving tracking result to cache...")
        # Save SCTC as JSON
        cache_sctc_file = cached_tracking_dir / f"tracking_max_age_{max_age}_sctc.json"
        sctc_tracked.write_json(cache_sctc_file)

        # Save statistics as JSON
        cache_stats_file = cached_tracking_dir / f"tracking_max_age_{max_age}_stats.json"
        stats_data = {
            "max_age": max_age,
            "num_trajectories": num_trajectories,
            "num_cells": num_cells,
            "avg_missing_rate": avg_missing_rate,
            "missing_rates": {},  # Don't save full missing_rates dict to keep file small
        }
        with open(cache_stats_file, "w") as f:
            json.dump(stats_data, f, indent=4)

        print(f"  Cached SCTC: {cache_sctc_file}")
        print(f"  Cached stats: {cache_stats_file}")

        # Store file paths for later use
        tracking_files[max_age] = {"sctc_file": cache_sctc_file, "stats_file": cache_stats_file, "cached": False}

        # Free memory - delete the SCTC
        del sctc_tracked

    # Save overall tracking summary
    summary_file = output_dir / "tracking_summary_all.json"
    with open(summary_file, "w") as f:
        json.dump(tracking_summary_all, f, indent=4)
    print(f"\nSaved tracking summary to: {summary_file}")

    return tracking_files, tracking_summary_all


def run_multiround_correction(cur_sctc, model, input_transforms, correction_out_dir, args, annotated_US_scs=[]):
    """
    Run multi-round correction on a single SCTC
    This is the original multi-round correction loop preserved intact
    """
    # Store original for comparison
    # Create a copy by creating a new SCTC and copying trajectories
    original_sctc = SingleCellTrajectoryCollection()
    original_sctc.track_id_to_trajectory = {k: v for k, v in cur_sctc.track_id_to_trajectory.items()}
    dist_to_boundary = args.dist_to_boundary

    # Multi-round correction implementation from original script
    visited_pairs = set()

    round_df_dict = {
        "round": [],
        "fix_attempt": [],
        "fixed_cases": [],
        "missing_rate": [],
        "filtered_missing_rate": [],
        "duplicate_underseg_fix_cases": [],
    }

    case_stats_df_dict = {
        "case_type": [],
        "extra_info": [],
        "state": [],
        "round": [],
    }

    # Set save_masks attribute for gen_missing_case_masks
    gen_missing_case_masks.save_masks = args.save_masks

    # Initialize corrected_sctc outside the loop in case no rounds are executed
    corrected_sctc = cur_sctc

    # Add initial statistics for the original SCTC (round 0)
    initial_missing_rates = compute_sctc_missing_rate(cur_sctc)
    initial_avg_missing_rate = np.mean(list(initial_missing_rates.values()))

    # Calculate filtered missing rate for initial SCTC
    filtered_initial_sctc = cur_sctc.filter_trajectories_by_length(min_length=30)
    filtered_initial_sctc = filter_boundary_traj(filtered_initial_sctc, dist=30)
    filtered_initial_missing_rates = compute_sctc_missing_rate(filtered_initial_sctc)
    filtered_initial_avg_missing_rate = np.mean(list(filtered_initial_missing_rates.values()))

    # Add round 0 statistics
    round_df_dict["round"].append(0)
    round_df_dict["fix_attempt"].append(0)
    round_df_dict["fixed_cases"].append(0)
    round_df_dict["duplicate_underseg_fix_cases"].append(0)
    round_df_dict["missing_rate"].append(initial_avg_missing_rate)
    round_df_dict["filtered_missing_rate"].append(filtered_initial_avg_missing_rate)

    # Save trajectory lengths for round 0 (initial state)
    save_trajectory_lengths_csv(cur_sctc, correction_out_dir / "trajectory_lengths_round_0.csv", round_num=0)

    print(f"\nInitial SCTC Statistics (Round 0):")
    print(f"  - Trajectories: {len(cur_sctc)}")
    print(f"  - Total cells: {len(cur_sctc.get_all_scs())}")
    print(f"  - Average missing rate: {initial_avg_missing_rate:.4f}")
    print(f"  - Filtered average missing rate: {filtered_initial_avg_missing_rate:.4f}")
    print("*" * 100)

    for round in range(1, args.max_round + 1):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" * 1)
        print("Round", round)
        print("# of visited pairs:", len(visited_pairs))
        print("# of cells in cur_sctc:", len(cur_sctc.get_all_scs()))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" * 1)

        # Add "sct" tmp key to each sc, for mapping sc to sct
        for tid, sct in cur_sctc:
            for sc in sct.get_all_scs():
                sc.tmp["sct"] = sct

        if args.minimal_output:
            # Use base directory instead of creating round directories
            round_out_dir = correction_out_dir
        else:
            round_out_dir = correction_out_dir / f"round_{round}"
            round_out_dir.mkdir(parents=True, exist_ok=True)

        res_dict = select_underseg_cells_by_missing(
            cur_sctc, threshold=args.match_threshold, search_interval=args.match_search_interval
        )
        prev_or_after_scs = res_dict["prev_or_after_scs"]
        underseg_candidate_pairs_by_missing = res_dict["underseg_candidate_pairs"]
        underseg_candidates_by_missing = set([pair[1] for pair in underseg_candidate_pairs_by_missing])

        print("Filtering out visited pairs...")
        pair_counter_before_filtering = len(underseg_candidate_pairs_by_missing)
        underseg_candidate_pairs_by_missing = [
            pair for pair in underseg_candidate_pairs_by_missing if pair not in visited_pairs
        ]

        print(
            "=====================================",
            "Missing logic summary",
            "=====================================",
        )

        all_scs = cur_sctc.get_all_scs()
        report_underseg_candidates(underseg_candidates_by_missing, annotated_US_scs, all_scs)

        print(
            "Number of pairs before Filtering out visited pairs:",
            pair_counter_before_filtering,
            "Number of pairs after Filtering out visited pairs:",
            len(underseg_candidate_pairs_by_missing),
        )
        visited_pairs = visited_pairs.union(set(underseg_candidate_pairs_by_missing))

        consecutive_underseg_scs_underseg_scs_not_in_gt = [
            _two_scs
            for _two_scs in underseg_candidate_pairs_by_missing
            if _two_scs[0].timeframe + 1 == _two_scs[1].timeframe
        ]
        print(
            "# of consecutive scs by missing logics that are not in predicted scs:",
            len(consecutive_underseg_scs_underseg_scs_not_in_gt),
        )
        print(
            "=====================================",
            "Missing logics end",
            "=====================================",
        )
        res_dict = select_underseg_cells_by_end(
            cur_sctc, threshold=args.match_threshold, search_interval=args.match_search_interval
        )
        end_scs = res_dict["end_cells"]
        underseg_candidate_pairs_by_ending = res_dict["underseg_candidate_pairs"]
        underseg_candidates_by_ending = set([pair[1] for pair in underseg_candidate_pairs_by_ending])

        print(
            "=====================================",
            "Ending logics begins",
            "=====================================",
        )

        report_underseg_candidates(underseg_candidates_by_ending, annotated_US_scs, all_scs)

        print("Filtering out visited pairs...")
        pair_counter_before_filtering = len(underseg_candidate_pairs_by_ending)
        underseg_candidate_pairs_by_ending = [
            pair for pair in underseg_candidate_pairs_by_ending if pair not in visited_pairs
        ]
        print(
            "Number of pairs before Filtering out visited pairs:",
            pair_counter_before_filtering,
            "Number of pairs after Filtering out visited pairs:",
            len(underseg_candidate_pairs_by_ending),
        )
        visited_pairs = visited_pairs.union(set(underseg_candidate_pairs_by_ending))

        end_consecutive_underseg_scs_underseg_scs_not_in_gt = [
            _two_scs
            for _two_scs in underseg_candidate_pairs_by_ending
            if _two_scs[0].timeframe + 1 == _two_scs[1].timeframe
        ]
        print(
            "# of consecutive scs by <END> logics that are not in predicted scs:",
            len(end_consecutive_underseg_scs_underseg_scs_not_in_gt),
        )
        print(
            "=====================================",
            "Ending logics complete",
            "=====================================",
        )

        print(
            "=====================================",
            "Missing + End Summary",
            "=====================================",
        )

        num_covered_by_logics = len(
            set(underseg_candidates_by_ending).union(set(underseg_candidates_by_missing)).intersection(annotated_US_scs)
        )

        print(
            "Percentage of TD-1 GT multimap cell found by logics",
            num_covered_by_logics / len(annotated_US_scs) if len(annotated_US_scs) > 0 else 0,
        )

        report_underseg_candidates(underseg_candidates_by_missing, annotated_US_scs, all_scs)
        print(
            "=====================================",
            "End",
            "=====================================",
        )

        # Skip visualization for large datasets to avoid memory issues
        if args.enable_viz and len(cur_sctc) < 100:  # Only plot for small datasets
            show_tracks_missing(
                cur_sctc.filter_trajectories_by_length(min_length=30),
                underseg_candidate_pairs_by_missing,
                figsize=(36, min(100, (5.0 / 30) * len(cur_sctc))),  # Cap height at 100
                y_interval=1,
            )
            # plt.savefig(round_out_dir / "before_correction_missing_track_plot.png")  # DISABLED for benchmark
            plt.close()
        else:
            print(f"[INFO] Skipping trajectory visualization for {len(cur_sctc)} trajectories (too large)")
        underseg_pairs_all = set(underseg_candidate_pairs_by_missing).union(underseg_candidate_pairs_by_ending)
        underseg_pairs_all = list(underseg_pairs_all)

        if len(underseg_pairs_all) == 0:
            print("[INFO] No underseg pairs to be fixed found in this round, exiting...")
            break

        # Generate and store masks in memory
        mask_cache = {}
        for underseg_candidate_pair in tqdm.tqdm(underseg_pairs_all, desc="Correcting masks"):
            in_sc, missing_sc = underseg_candidate_pair[0], underseg_candidate_pair[1]
            _in_watershed, _missing_watershed = gen_missing_case_masks(
                in_sc,
                missing_sc,
                model,
                input_transforms,
                out_dir=correction_out_dir,  # NOT round_out_dir
                padding=20,
                h_threshold=args.h_threshold,
            )
            # Store masks in cache
            mask_cache[(in_sc.timeframe, in_sc.id)] = _in_watershed
            mask_cache[(missing_sc.timeframe, missing_sc.id)] = _missing_watershed

        fixed_attempt_counter = 0
        case_types = []
        fixed_result_dicts = []
        example_cases_saved = 0
        max_examples = 5  # Save first 5 examples per round

        for idx, underseg_candidate_pair in enumerate(tqdm.tqdm(underseg_pairs_all, desc="Fixing by logics")):
            assert underseg_candidate_pair[0] in cur_sctc.get_all_scs(), "Missing sc: {}".format(
                underseg_candidate_pair[0].id
            )

            res = fix_missing_trajectory(
                underseg_candidate_pair[0],
                underseg_candidate_pair[1],
                padding=args.padding,
                area_threshold=args.area_threshold,
                model=model,
                input_transforms=input_transforms,
                h_threshold=args.h_threshold,
                out_threshold=args.out_threshold,
            )
            fixed_attempt_counter += 1
            case_types.append(res["case_type"])
            fixed_result_dicts.append(res)

            # Save example visualizations for the first few cases
            if args.enable_viz and example_cases_saved < max_examples and res["state"] == "fixed":
                fig_dir = correction_out_dir / "correction_figs"
                fig_dir.mkdir(parents=True, exist_ok=True)

                sc1, sc2 = underseg_candidate_pair[0], underseg_candidate_pair[1]

                # Create a simple visualization showing the correction
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                # Show original undersegmented cell
                axes[0].imshow(sc1.get_img(), cmap="gray")
                axes[0].contour(sc1.get_mask(), colors="red", linewidths=2)
                axes[0].set_title(f"Original Cell\nTime: {sc1.timeframe}")
                axes[0].axis("off")

                # Show missing cell location
                axes[1].imshow(sc2.get_img(), cmap="gray")
                axes[1].contour(sc2.get_mask(), colors="blue", linewidths=2)
                axes[1].set_title(f"Missing Cell\nTime: {sc2.timeframe}")
                axes[1].axis("off")

                # Show case type and result
                axes[2].text(
                    0.5,
                    0.5,
                    f'Case Type: {res["case_type"]}\nRound: {round}\nFixed: {res["state"]}',
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=axes[2].transAxes,
                )
                axes[2].axis("off")

                plt.suptitle(f"Correction Example {example_cases_saved + 1}")
                plt.tight_layout()
                plt.savefig(fig_dir / f'round{round}_example{example_cases_saved + 1}_{res["case_type"]}.png', dpi=150)
                plt.close()

                example_cases_saved += 1

        # Plot case types distribution
        print("[Start] Distribution of case types")

        # Calculate case type counts
        case_type_counts = {}
        for case_type in case_types:
            case_type_counts[case_type] = case_type_counts.get(case_type, 0) + 1

        # Print distribution
        print(f"\nRound {round} Case Type Distribution:")
        print("=" * 50)
        for case_type, count in sorted(case_type_counts.items()):
            percentage = (count / len(case_types) * 100) if case_types else 0
            print(f"{case_type}: {count} ({percentage:.1f}%)")
        print(f"Total cases: {len(case_types)}")
        print("=" * 50)

        if args.enable_viz:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
            sns.set_theme(style="whitegrid", palette="pastel")
            sns.histplot(
                case_types,
                kde=False,
                color="blue",
                edgecolor="black",
                ax=ax,
            )

            plt.title(f"Distribution of Case Types - Round {round}", fontsize=16, fontweight="bold")
            plt.xlabel("Case Type", fontsize=14, fontweight="bold")
            plt.ylabel("Count", fontsize=14, fontweight="bold")

            plt.xticks(fontsize=12, rotation=45, ha="right")
            plt.yticks(fontsize=12)

            # Add count labels on bars
            for patch in ax.patches:
                height = patch.get_height()
                if height > 0:
                    ax.text(patch.get_x() + patch.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom")

            fig.tight_layout()
            plt.savefig(round_out_dir / f"case_type_dist_round_{round}.png", dpi=300)
            plt.close()

        print("[Complete] Distribution of case types")

        corrected_sctc = SingleCellTrajectoryCollection()
        for _, sct in cur_sctc:
            corrected_sctc.add_trajectory(sct)

        print("Constructed a new sctc with {} trajectories".format(len(corrected_sctc)))

        assert len(underseg_pairs_all) == len(
            fixed_result_dicts
        ), "Length mismatch between underseg_pairs_all and fixed_result_dicts, check the code"

        fixed_case_counter = 0
        duplicate_underseg_fix_counter = 0
        for idx in range(len(underseg_pairs_all)):
            underseg_candidate_pair = underseg_pairs_all[idx]
            fix_res_dict = fixed_result_dicts[idx]
            case_stats_df_dict["case_type"].append(fix_res_dict["case_type"])
            case_stats_df_dict["extra_info"].append(
                {
                    "sc1": str(underseg_candidate_pair[0].id),
                    "sc2": str(underseg_candidate_pair[1].id),
                }
            )
            case_stats_df_dict["state"].append(fix_res_dict["state"])
            case_stats_df_dict["round"].append(round)

            if fix_res_dict["state"] == "skipped":
                continue
            elif fix_res_dict["state"] == "fixed":
                assert underseg_candidate_pair[0] == fix_res_dict["sc_pair"][0]
                fixed_case_counter += 1
                # Important: Do not use sct stored in tmp directly
                # Because it may not be the latest sct in cur_sctc
                sc1, sc2 = fix_res_dict["sc_pair"]
                useg_timeframe = sc2.timeframe
                track_id = underseg_candidate_pair[0].tmp["sct"].track_id
                traj_in_sctc = corrected_sctc[track_id]
                _updated_sct1 = overwrite_sct(traj_in_sctc, fix_res_dict["updated_scts"][0], useg_timeframe)
                corrected_sctc.pop_trajectory_by_id(track_id)
                corrected_sctc.add_trajectory(_updated_sct1)
                assert (
                    abs(sc2.timeframe - sc1.timeframe) <= args.match_search_interval
                ), f"Timeframe mismatch: {sc1.timeframe} vs. {sc2.timeframe}; diff: {abs(sc2.timeframe - sc1.timeframe)}, match_search_interval: {args.match_search_interval}"
                assert len(_updated_sct1) >= len(traj_in_sctc)
                assert (
                    _updated_sct1.get_time_span_length()
                    <= traj_in_sctc.get_time_span_length() + args.match_search_interval
                ), f"span length update error: {_updated_sct1.get_time_span_length()} != {traj_in_sctc.get_time_span_length() + args.match_search_interval}, # scs: {len(_updated_sct1.get_all_scs())} vs. {len(traj_in_sctc.get_all_scs())}, sc2: {sc2.timeframe}, sc1: {sc1.timeframe}, updated_sct1 times:{_updated_sct1.times}, traj_in_sctc times: {traj_in_sctc.times}"
                underseg_candidate_pair[0].tmp["sct"] = _updated_sct1
                if len(_updated_sct1) == len(traj_in_sctc):
                    duplicate_underseg_fix_counter += 1
                # Update the second sc
                assert len(fix_res_dict["updated_scts"]) > 1, "Missing second sct in fix_res_dict"
                if len(fix_res_dict["updated_scts"]) > 1:
                    second_track_id = underseg_candidate_pair[1].tmp["sct"].track_id
                    second_traj_in_sctc = corrected_sctc[second_track_id]
                    _updated_sct_second = overwrite_sct(
                        second_traj_in_sctc, fix_res_dict["updated_scts"][1], useg_timeframe
                    )
                    corrected_sctc.pop_trajectory_by_id(second_track_id)
                    corrected_sctc.add_trajectory(_updated_sct_second)
                    assert len(_updated_sct_second) >= len(second_traj_in_sctc)
                    assert (
                        _updated_sct_second.get_time_span_length() == second_traj_in_sctc.get_time_span_length()
                    ), f"{_updated_sct_second.get_time_span_length()} != {second_traj_in_sctc.get_time_span_length()}"
                    underseg_candidate_pair[1].tmp["sct"] = _updated_sct_second

        print("*" * 100)
        print(f"* Round-{round} Summary")

        print(
            "Corrected SCTC: {} trajectories, Original SCTC: {} trajectories".format(len(corrected_sctc), len(cur_sctc))
        )
        print("# of Total cells in corrected SCTC:", len(corrected_sctc.get_all_scs()))
        print("# of Total cells in round-start SCTC:", len(cur_sctc.get_all_scs()))
        print("# Fixed case:", fixed_case_counter)
        print(
            "Number of fix attempts by CSN in this round:",
            fixed_attempt_counter,
        )
        print(
            "Percentage of cases fixed by CSN in this round:",
            fixed_case_counter / fixed_attempt_counter if fixed_attempt_counter > 0 else 0,
        )

        round_df_dict["round"].append(round)
        round_df_dict["fix_attempt"].append(fixed_attempt_counter)
        round_df_dict["fixed_cases"].append(fixed_case_counter)
        round_df_dict["duplicate_underseg_fix_cases"].append(duplicate_underseg_fix_counter)

        round_df_dict["missing_rate"].append(np.array(list(compute_sctc_missing_rate(corrected_sctc).values())).mean())
        filtered_corrected_sctc = corrected_sctc.filter_trajectories_by_length(min_length=30)
        filtered_corrected_sctc = filter_boundary_traj(filtered_corrected_sctc, dist=30)

        round_df_dict["filtered_missing_rate"].append(
            np.array(list(compute_sctc_missing_rate(filtered_corrected_sctc).values())).mean()
        )

        # Save trajectory lengths for this round
        save_trajectory_lengths_csv(
            corrected_sctc, correction_out_dir / f"trajectory_lengths_round_{round}.csv", round_num=round
        )

        print("*" * 100)
        cur_sctc = corrected_sctc

        # Skip large histograms for big datasets
        if args.enable_viz:
            # Save trajectory length histograms for each round
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

            # Original trajectory lengths
            filtered_original_sctc = filter_boundary_traj(original_sctc, dist=dist_to_boundary)
            filtered_original_sctc.histogram_traj_length(ax=ax1)
            ax1.set_title(f"Original Trajectory Lengths - Round {round}")
            ax1.set_xlabel("Trajectory Length")
            ax1.set_ylabel("Count")
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

            # Corrected trajectory lengths
            corrected_sctc.histogram_traj_length(ax=ax2, color="blue")
            ax2.set_title(f"Corrected Trajectory Lengths - Round {round}")
            ax2.set_xlabel("Trajectory Length")
            ax2.set_ylabel("Count")
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

            plt.tight_layout()
            plt.savefig(round_out_dir / f"trajectory_length_histogram_round_{round}.png", dpi=150)
            plt.close()

            print(f"[INFO] Saved trajectory length histograms for round {round}")

        round_df = pd.DataFrame(round_df_dict)
        case_stats_df = pd.DataFrame(case_stats_df_dict)

        round_df.to_csv(correction_out_dir / "round_df.csv", index=False)
        case_stats_df.to_csv(correction_out_dir / "case_stats_df.csv", index=False)

    # Save final corrected SCTC
    corrected_sctc.write_json(
        correction_out_dir / (f"corrected_sctc-final-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json")
    )

    # Save final trajectory lengths
    final_round = len(round_df_dict["round"]) - 1  # -1 because round_df_dict includes round 0
    save_trajectory_lengths_csv(
        corrected_sctc, correction_out_dir / f"trajectory_lengths_final.csv", round_num=final_round
    )

    return corrected_sctc, round_df_dict, case_stats_df_dict


def main():
    args = get_parser().parse_args()

    # Setup paths
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    lcx_data_dir = Path(args.lcx_data_dir)
    if not lcx_data_dir.exists():
        raise ValueError(f"Data directory does not exist: {lcx_data_dir}")

    # Create experiment directory
    exp_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"multiround_tracking_benchmark_{exp_timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory: {exp_dir}")

    # Save experiment configuration
    config = vars(args)
    config["timestamp"] = exp_timestamp
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Load model
    print("\nLoading CSN model...")
    model = CorrectSegNetAux.load_from_checkpoint(args.model_ckpt)
    model.cuda()
    model.eval()
    input_transforms = CustomTransformEdtV9(
        degrees=0, shear=0, flip_p=0, use_gaussian_blur=True, gaussian_blur_sigma=15
    )

    # In DEBUG mode, limit the number of cells
    if args.DEBUG:
        print("DEBUG mode: Loading limited subset of cells")
        all_scs = []  # Will be filled below with limited cells
    else:
        pass  # all_scs already initialized as empty list

    # Load cells
    if args.run_cxa_experiment:
        print("\nRunning CXA experiment in LivecellX paper...")
        lcx_out_dir = Path("./notebook_results/CXA_process2_7_19/")
        try:
            cp_all_scs = SingleCellStatic.load_single_cells_json(lcx_out_dir / "single_cells_zero_based_time.json")
            expert_filed_scs = SingleCellTrajectoryCollection.load_from_json_file(
                lcx_out_dir / "Sijie-labeled-missing_cells-9-29.json"
            ).get_all_scs()
            all_scs = cp_all_scs + expert_filed_scs
            print(f"Loaded {len(cp_all_scs)} cells from CXA + {len(expert_filed_scs)} expert-labeled cells")
            print(f"Total: {len(all_scs)} cells")
        except FileNotFoundError as e:
            assert False, f"CXA data files not found: {e}. Please ensure the CXA experiment files are available."

    # In DEBUG mode, limit cells
    if args.DEBUG and len(all_scs) > 1000:
        print(f"DEBUG: Limiting cells from {len(all_scs)} to 1000")
        all_scs = all_scs[:1000]

    # Load annotation if available
    annotated_US_scs = []
    annotation_path = lcx_data_dir / args.annotation_file
    if annotation_path.exists():
        print("\nLoading annotations...")
        annotation_table = pd.read_csv(annotation_path)

        def in_annotation_US(sc: SingleCellStatic):
            return (
                str(sc.id) in annotation_table["sc_id"].values
                and annotation_table[annotation_table["sc_id"] == str(sc.id)]["label"].values[0] == "US"
            )

        annotated_US_scs = [sc for sc in all_scs if in_annotation_US(sc)]
        print(f"Found {len(annotated_US_scs)} annotated undersegmented cells")

    ###########################################################################
    # TRACKING BENCHMARK OR LOAD PRE-EXISTING SCTC
    ###########################################################################

    if args.skip_tracking:
        if not args.input_sctc:
            raise ValueError("--input_sctc must be specified when using --skip_tracking")

        print("\n" + "=" * 80)
        print("Skipping tracking phase - using pre-existing SCTC file")
        print(f"Loading SCTC from: {args.input_sctc}")
        print("=" * 80)

        # Load the pre-existing SCTC
        sctc = SingleCellTrajectoryCollection.load_from_json_file(args.input_sctc)
        print(f"Loaded {len(sctc)} trajectories with {len(sctc.get_all_scs())} cells")

        # Calculate missing rates
        missing_rates = compute_sctc_missing_rate(sctc)
        avg_missing_rate = np.mean(list(missing_rates.values()))

        # Create a mock tracking_files entry for the correction phase
        tracking_dir = exp_dir / "tracking_benchmark"
        tracking_dir.mkdir(exist_ok=True)

        # Save the SCTC to the tracking directory
        sctc_file = tracking_dir / "input_sctc.json"
        sctc.write_json(sctc_file)

        # Save stats
        stats = {
            "num_trajectories": len(sctc),
            "num_cells": len(sctc.get_all_scs()),
            "avg_missing_rate": avg_missing_rate,
        }
        stats_file = tracking_dir / "input_sctc_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=4)

        # Create tracking_files dictionary with a single entry
        tracking_files = {
            0: {  # Use 0 as a special marker for pre-loaded SCTC
                "sctc_file": sctc_file,
                "stats_file": stats_file,
            }
        }

        tracking_summary_all = [
            {
                "max_age": 0,
                "num_trajectories": len(sctc),
                "num_cells": len(sctc.get_all_scs()),
                "avg_missing_rate": avg_missing_rate,
            }
        ]

    elif args.DEBUG:
        print("\n" + "=" * 80)
        print("DEBUG mode: Running limited tracking for faster testing...")
        print("=" * 80)

        tracking_dir = exp_dir / "tracking_benchmark"
        tracking_dir.mkdir(exist_ok=True)

        # Run tracking on limited cells with cache disabled for debug
        tracking_files, tracking_summary_all = run_tracking_benchmark(
            all_scs,
            args.max_ages,
            args.min_hits,
            tracking_dir,
            force_retrack=args.force_retrack,  # Use command line argument
            args=args,
        )
    else:
        print("\n" + "=" * 80)
        print("Running tracking benchmark with different max_age parameters...")
        print("=" * 80)

        tracking_dir = exp_dir / "tracking_benchmark"
        tracking_dir.mkdir(exist_ok=True)

        tracking_files, tracking_summary_all = run_tracking_benchmark(
            all_scs, args.max_ages, args.min_hits, tracking_dir, args.force_retrack, args=args
        )

    # Save tracking summary
    if len(tracking_files) > 0 and len(tracking_summary_all) > 0:
        # Ensure we have data for all max_ages in tracking_files
        summary_dict = {s["max_age"]: s for s in tracking_summary_all}

        # Create lists with matching lengths
        max_ages = sorted(list(tracking_files.keys()))
        num_trajectories = []
        num_cells = []
        avg_missing_rates = []

        for max_age in max_ages:
            if max_age in summary_dict:
                num_trajectories.append(summary_dict[max_age]["num_trajectories"])
                num_cells.append(summary_dict[max_age]["num_cells"])
                avg_missing_rates.append(summary_dict[max_age]["avg_missing_rate"])
            else:
                # This shouldn't happen, but handle it gracefully
                print(f"Warning: No summary found for max_age={max_age}")
                num_trajectories.append(0)
                num_cells.append(0)
                avg_missing_rates.append(0)

        tracking_summary = pd.DataFrame(
            {
                "max_age": max_ages,
                "num_trajectories": num_trajectories,
                "num_cells": num_cells,
                "avg_missing_rate": avg_missing_rates,
            }
        )
        tracking_summary.to_csv(tracking_dir / "tracking_summary.csv", index=False)
    else:
        print("Warning: No tracking results to summarize")

    # Plot tracking results
    if args.enable_viz:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(tracking_summary["max_age"], tracking_summary["num_trajectories"], "o-")
        ax1.set_xlabel("Max Age")
        ax1.set_ylabel("Number of Trajectories")
        ax1.set_xscale("log")
        ax1.set_title("Number of Trajectories vs Max Age")
        ax1.grid(True)

        ax2.plot(tracking_summary["max_age"], tracking_summary["avg_missing_rate"], "o-")
        ax2.set_xlabel("Max Age")
        ax2.set_ylabel("Average Missing Rate")
        ax2.set_xscale("log")
        ax2.set_title("Average Missing Rate vs Max Age")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(tracking_dir / "tracking_analysis.png", dpi=300)
        plt.close()
    else:
        print("[INFO] Skipping tracking analysis plots")

    ###########################################################################
    # MULTI-ROUND CORRECTION FOR EACH TRACKING RESULT
    ###########################################################################

    print("\n" + "=" * 80)
    print("TRACKING PHASE COMPLETE - Starting correction phase...")
    print("=" * 80)
    print(f"Will run multi-round correction for {len(tracking_files)} tracking results")
    print(f"Max ages to process: {list(tracking_files.keys())}")
    print("=" * 80)

    correction_summary = []

    for max_age, tracking_file_info in tracking_files.items():
        print(f"\n{'='*60}")
        print(f"Processing max_age={max_age}")
        print(f"{'='*60}")

        # Create output directory for this max_age
        correction_dir = exp_dir / f"correction_max_age_{max_age}"
        correction_dir.mkdir(exist_ok=True)

        # Load SCTC from file for this tracking result
        print(f"Loading SCTC from {tracking_file_info['sctc_file']}...")
        sctc = SingleCellTrajectoryCollection.load_from_json_file(tracking_file_info["sctc_file"])

        # Load stats for this tracking result
        with open(tracking_file_info["stats_file"], "r") as f:
            tracking_stats = json.load(f)

        print(f"Loaded {len(sctc)} trajectories with {len(sctc.get_all_scs())} cells")

        # Filter by boundary
        sctc = filter_boundary_traj(sctc, dist=args.dist_to_boundary)
        print(f"After boundary filtering: {len(sctc)} trajectories")

        # Save hyperparameters
        hyperparams = {
            "max_age": max_age,
            "match_threshold": args.match_threshold,
            "match_search_interval": args.match_search_interval,
            "max_round": args.max_round,
            "h_threshold": args.h_threshold,
            "out_threshold": args.out_threshold,
            "padding": args.padding,
            "area_threshold": args.area_threshold,
            "dist_to_boundary": args.dist_to_boundary,
        }

        with open(correction_dir / "hyperparams.json", "w") as f:
            json.dump(hyperparams, f, indent=4)

        # Run multi-round correction
        corrected_sctc, round_stats, case_stats = run_multiround_correction(
            sctc, model, input_transforms, correction_dir, args, annotated_US_scs
        )

        # Calculate final statistics
        final_missing_rates = compute_sctc_missing_rate(corrected_sctc)
        final_avg_missing_rate = np.mean(list(final_missing_rates.values()))

        correction_summary.append(
            {
                "max_age": max_age,
                "initial_trajectories": tracking_stats["num_trajectories"],
                "initial_missing_rate": tracking_stats["avg_missing_rate"],
                "final_trajectories": len(corrected_sctc),
                "final_missing_rate": final_avg_missing_rate,
                "missing_rate_reduction": tracking_stats["avg_missing_rate"] - final_avg_missing_rate,
                "total_rounds": len(round_stats["round"]) if round_stats["round"] else 0,
                "total_fixes": sum(round_stats["fixed_cases"]) if round_stats["fixed_cases"] else 0,
            }
        )

        # Plot missing rate evolution for this max_age
        if args.enable_viz and round_stats["round"]:
            round_df = pd.DataFrame(round_stats)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot(round_df["round"], round_df["missing_rate"], "o-", label="All trajectories")
            ax.plot(round_df["round"], round_df["filtered_missing_rate"], "s-", label="Filtered trajectories")
            ax.set_xlabel("Round")
            ax.set_ylabel("Average Missing Rate")
            ax.set_title(f"Missing Rate Evolution (max_age={max_age})")
            ax.legend()
            ax.grid(True)
            plt.savefig(correction_dir / "missing_rate_evolution.png", dpi=300)
            plt.close()

        # Save individual max_age summary immediately after completion
        max_age_summary = {
            "max_age": max_age,
            "initial_trajectories": tracking_stats["num_trajectories"],
            "initial_missing_rate": tracking_stats["avg_missing_rate"],
            "final_trajectories": len(corrected_sctc),
            "final_missing_rate": final_avg_missing_rate,
            "missing_rate_reduction": tracking_stats["avg_missing_rate"] - final_avg_missing_rate,
            "total_rounds": len(round_stats["round"]) if round_stats["round"] else 0,
            "total_fixes": sum(round_stats["fixed_cases"]) if round_stats["fixed_cases"] else 0,
            "completion_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save this max_age's summary
        with open(correction_dir / "max_age_summary.json", "w") as f:
            json.dump(max_age_summary, f, indent=4)

        # Save case statistics for this max_age
        if case_stats:
            case_stats_df = pd.DataFrame(case_stats)
            case_stats_df.to_csv(correction_dir / "case_statistics.csv", index=False)

            # Create case type summary
            case_type_summary = case_stats_df.groupby(["round", "case_type", "state"]).size().reset_index(name="count")
            case_type_summary.to_csv(correction_dir / "case_type_summary.csv", index=False)

            if args.enable_viz:
                # Plot case type distribution across rounds
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                case_type_by_round = (
                    case_stats_df[case_stats_df["state"] == "fixed"]
                    .groupby(["round", "case_type"])
                    .size()
                    .unstack(fill_value=0)
                )
                case_type_by_round.plot(kind="bar", stacked=True, ax=ax)
                ax.set_xlabel("Round")
                ax.set_ylabel("Number of Fixed Cases")
                ax.set_title(f"Fixed Cases by Type Across Rounds (max_age={max_age})")
                ax.legend(title="Case Type", bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()
                plt.savefig(correction_dir / "fixed_cases_by_type_across_rounds.png", dpi=300)
                plt.close()

        # Also save current overall progress
        correction_summary_df = pd.DataFrame(correction_summary)
        correction_summary_df.to_csv(exp_dir / "correction_summary_progress.csv", index=False)
        print(f"\n✅ Completed max_age={max_age} - Results saved to {correction_dir}")

        # Clean up memory
        del sctc
        del corrected_sctc
        import gc

        gc.collect()

    # Save overall correction summary
    correction_summary_df = pd.DataFrame(correction_summary)
    correction_summary_df.to_csv(exp_dir / "correction_summary.csv", index=False)

    # Plot overall comparison
    if args.enable_viz and not correction_summary_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Initial vs final trajectories
        ax = axes[0, 0]
        x = np.arange(len(correction_summary_df))
        width = 0.35
        ax.bar(x - width / 2, correction_summary_df["initial_trajectories"], width, label="Initial")
        ax.bar(x + width / 2, correction_summary_df["final_trajectories"], width, label="Final")
        ax.set_xlabel("Max Age")
        ax.set_ylabel("Number of Trajectories")
        ax.set_title("Trajectories Before and After Correction")
        ax.set_xticks(x)
        ax.set_xticklabels(correction_summary_df["max_age"])
        ax.legend()

        # Initial vs final missing rates
        ax = axes[0, 1]
        ax.bar(x - width / 2, correction_summary_df["initial_missing_rate"], width, label="Initial")
        ax.bar(x + width / 2, correction_summary_df["final_missing_rate"], width, label="Final")
        ax.set_xlabel("Max Age")
        ax.set_ylabel("Average Missing Rate")
        ax.set_title("Missing Rate Before and After Correction")
        ax.set_xticks(x)
        ax.set_xticklabels(correction_summary_df["max_age"])
        ax.legend()

        # Missing rate reduction
        ax = axes[1, 0]
        ax.bar(x, correction_summary_df["missing_rate_reduction"])
        ax.set_xlabel("Max Age")
        ax.set_ylabel("Missing Rate Reduction")
        ax.set_title("Missing Rate Improvement by Max Age")
        ax.set_xticks(x)
        ax.set_xticklabels(correction_summary_df["max_age"])

        # Total fixes
        ax = axes[1, 1]
        ax.bar(x, correction_summary_df["total_fixes"])
        ax.set_xlabel("Max Age")
        ax.set_ylabel("Total Fixes")
        ax.set_title("Total Corrections by Max Age")
        ax.set_xticks(x)
        ax.set_xticklabels(correction_summary_df["max_age"])

        plt.tight_layout()
        plt.savefig(exp_dir / "overall_comparison.png", dpi=300)
        plt.close()
    else:
        print("[INFO] Skipping overall comparison plots")

    print("\n" + "=" * 80)
    print("Experiment completed!")
    print(f"Results saved to: {exp_dir}")
    print("=" * 80)

    # Print summary
    print("\nCorrection Summary:")
    if correction_summary_df.empty:
        print("WARNING: No correction results found!")
        print("Possible reasons:")
        print("1. No tracking results were generated")
        print("2. Correction phase was skipped or failed")
        print("3. All SCTCs failed to load")
        print(f"\nNumber of tracking files: {len(tracking_files)}")
        if len(tracking_files) > 0:
            print(f"Max ages attempted: {list(tracking_files.keys())}")
    else:
        print(correction_summary_df.to_string())

        # Find best max_age based on final missing rate
        best_idx = correction_summary_df["final_missing_rate"].idxmin()
        best_max_age = correction_summary_df.loc[best_idx, "max_age"]
        print(f"\nBest max_age based on final missing rate: {best_max_age}")
        print(f"  Final missing rate: {correction_summary_df.loc[best_idx, 'final_missing_rate']:.4f}")
        print(f"  Missing rate reduction: {correction_summary_df.loc[best_idx, 'missing_rate_reduction']:.4f}")


if __name__ == "__main__":
    main()
