"""Generic LiveCellX post-SORT trajectory repair before CS-Net correction.

The two patches reproduce the order used in
``step5_track_and_napari_correction copy.ipynb``:

1. Restore segmented cells that SORT did not place in any trajectory. Missing
   cells are chained across adjacent frames and prepended to the best future
   trajectory by mask IoU, or retained as a new trajectory when no target is
   available.
2. Remove spatially duplicated short trajectories and reconnect non-overlapping
   short trajectory fragments using forward-first mask-IoU matching.

Both operations mutate and return the supplied trajectory collection. They must
run after SORT and before CS-Net and TimeSformer.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

from livecellx.core import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection


def _time_index(sctc: SingleCellTrajectoryCollection) -> Dict[int, list]:
    index: Dict[int, list] = {}
    for tid, traj in sctc:
        for timeframe, sc in traj:
            index.setdefault(int(timeframe), []).append((int(tid), sc))
    return index


def _best_iou(sc, candidates, exclude_tid=None, min_iou=0.0):
    best = None
    for tid, candidate in candidates:
        if exclude_tid is not None and int(tid) == int(exclude_tid):
            continue
        iou = float(sc.compute_iou(candidate))
        if iou > min_iou and (best is None or iou > best[0]):
            best = (iou, int(tid), candidate)
    return best


def _bbox_geometry(sc) -> Tuple[float, float, float]:
    bbox = [float(value) for value in sc.bbox]
    height = max(1.0, bbox[2] - bbox[0])
    width = max(1.0, bbox[3] - bbox[1])
    center_y = (bbox[0] + bbox[2]) / 2.0
    center_x = (bbox[1] + bbox[3]) / 2.0
    return center_y, center_x, math.hypot(height, width)


def _continuity_score(previous, current) -> Tuple[float, dict]:
    """Score adjacent cells using overlap, motion, and bbox-area stability."""
    iou = float(previous.compute_iou(current))
    prev_y, prev_x, prev_scale = _bbox_geometry(previous)
    cur_y, cur_x, cur_scale = _bbox_geometry(current)
    distance = math.hypot(cur_y - prev_y, cur_x - prev_x)
    scale = max(1.0, (prev_scale + cur_scale) / 2.0)
    distance_score = math.exp(-distance / scale)

    prev_bbox = [float(value) for value in previous.bbox]
    cur_bbox = [float(value) for value in current.bbox]
    prev_area = max(1.0, (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1]))
    cur_area = max(1.0, (cur_bbox[2] - cur_bbox[0]) * (cur_bbox[3] - cur_bbox[1]))
    area_score = math.exp(-abs(math.log(cur_area / prev_area)))
    score = 0.55 * iou + 0.30 * distance_score + 0.15 * area_score
    return float(score), {
        "iou": float(iou),
        "center_distance": float(distance),
        "distance_score": float(distance_score),
        "area_score": float(area_score),
    }


def _continuity_score_upper_bound(previous, current) -> float:
    """Cheap upper bound used only to reject impossible candidates."""
    prev_y, prev_x, prev_scale = _bbox_geometry(previous)
    cur_y, cur_x, cur_scale = _bbox_geometry(current)
    distance = math.hypot(cur_y - prev_y, cur_x - prev_x)
    scale = max(1.0, (prev_scale + cur_scale) / 2.0)
    distance_score = math.exp(-distance / scale)

    prev_bbox = [float(value) for value in previous.bbox]
    cur_bbox = [float(value) for value in current.bbox]
    prev_area = max(1.0, (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1]))
    cur_area = max(1.0, (cur_bbox[2] - cur_bbox[0]) * (cur_bbox[3] - cur_bbox[1]))
    area_score = math.exp(-abs(math.log(cur_area / prev_area)))

    bboxes_overlap = (
        prev_bbox[0] <= cur_bbox[2]
        and prev_bbox[2] >= cur_bbox[0]
        and prev_bbox[1] <= cur_bbox[3]
        and prev_bbox[3] >= cur_bbox[1]
    )
    iou_upper = 1.0 if bboxes_overlap else 0.0
    return 0.55 * iou_upper + 0.30 * distance_score + 0.15 * area_score


def repair_identity_switches(
    sctc: SingleCellTrajectoryCollection,
    max_frame_gap: int = 1,
    minimum_candidate_score: float = 0.45,
    minimum_improvement: float = 0.15,
    maximum_current_score: float = 0.65,
    max_repairs: int = 10000,
) -> dict:
    """Repair an established track that jumps as a new nearby track begins.

    A repair swaps the complete suffixes of the established and newly started
    trajectories. This preserves every cell and track ID while assigning the
    established ID to the spatially continuous cell. Only conservative,
    one-frame-boundary events with a clear score improvement are accepted.
    """
    repairs = []
    examined_boundaries = 0
    used_track_ids_by_time = {}

    # Track start times do not change when two suffixes are swapped: the
    # established track keeps its prefix and the newly started track still
    # begins at switch_time. Build this index once instead of rebuilding a
    # full cell-by-time index after every accepted repair.
    starting_track_ids = {}
    for track_id, trajectory in sctc:
        start_time = int(trajectory.get_timeframe_span()[0])
        starting_track_ids.setdefault(start_time, []).append(int(track_id))

    # Suffix swaps move SingleCellStatic objects between trajectories but do
    # not alter their masks or bounding boxes. Therefore a continuity score
    # for the same ordered pair is invariant and can be reused exactly.
    score_cache = {}

    def cached_continuity_score(previous, current):
        key = (id(previous), id(current))
        result = score_cache.get(key)
        if result is None:
            result = _continuity_score(previous, current)
            score_cache[key] = result
        return result

    while len(repairs) < int(max_repairs):
        proposals = []
        for primary_tid, primary in sctc:
            primary_cells = primary.get_sorted_scs()
            for index in range(1, len(primary_cells)):
                previous = primary_cells[index - 1]
                current = primary_cells[index]
                switch_time = int(current.timeframe)
                gap = switch_time - int(previous.timeframe)
                if gap <= 0 or gap > int(max_frame_gap):
                    continue
                used_at_time = used_track_ids_by_time.get(switch_time, set())
                if int(primary_tid) in used_at_time:
                    continue
                examined_boundaries += 1
                candidate_cells = []
                for candidate_tid in starting_track_ids.get(switch_time, []):
                    if int(candidate_tid) == int(primary_tid):
                        continue
                    if int(candidate_tid) in used_at_time:
                        continue
                    candidate = sctc.get_trajectory(int(candidate_tid))
                    candidate_cell = candidate.get_sc(switch_time)
                    if (
                        _continuity_score_upper_bound(previous, candidate_cell)
                        < float(minimum_candidate_score)
                    ):
                        continue
                    candidate_cells.append((int(candidate_tid), candidate_cell))
                if not candidate_cells:
                    continue
                current_score, current_details = cached_continuity_score(previous, current)
                if current_score > float(maximum_current_score):
                    continue

                for candidate_tid, candidate_cell in candidate_cells:
                    candidate_score, candidate_details = cached_continuity_score(
                        previous, candidate_cell
                    )
                    improvement = candidate_score - current_score
                    if candidate_score < float(minimum_candidate_score):
                        continue
                    if improvement < float(minimum_improvement):
                        continue
                    proposals.append(
                        (
                            float(improvement),
                            float(candidate_score),
                            int(primary_tid),
                            int(candidate_tid),
                            switch_time,
                            float(current_score),
                            current_details,
                            candidate_details,
                        )
                    )

        if not proposals:
            break

        proposals.sort(key=lambda item: (-item[0], -item[1], item[4], item[2], item[3]))
        (
            improvement,
            candidate_score,
            primary_tid,
            candidate_tid,
            switch_time,
            current_score,
            current_details,
            candidate_details,
        ) = proposals[0]
        primary = sctc.get_trajectory(primary_tid)
        candidate = sctc.get_trajectory(candidate_tid)

        primary_prefix = {
            int(time): sc
            for time, sc in primary.timeframe_to_single_cell.items()
            if int(time) < switch_time
        }
        primary_suffix = {
            int(time): sc
            for time, sc in primary.timeframe_to_single_cell.items()
            if int(time) >= switch_time
        }
        candidate_prefix = {
            int(time): sc
            for time, sc in candidate.timeframe_to_single_cell.items()
            if int(time) < switch_time
        }
        candidate_suffix = {
            int(time): sc
            for time, sc in candidate.timeframe_to_single_cell.items()
            if int(time) >= switch_time
        }
        if not primary_suffix or not candidate_suffix:
            break

        primary.timeframe_to_single_cell = {**primary_prefix, **candidate_suffix}
        candidate.timeframe_to_single_cell = {**candidate_prefix, **primary_suffix}
        used_track_ids_by_time.setdefault(switch_time, set()).update(
            {int(primary_tid), int(candidate_tid)}
        )
        repairs.append(
            {
                "primary_track_id": int(primary_tid),
                "new_track_id": int(candidate_tid),
                "switch_time": int(switch_time),
                "current_score": float(current_score),
                "replacement_score": float(candidate_score),
                "score_improvement": float(improvement),
                "current_transition": current_details,
                "replacement_transition": candidate_details,
                "primary_suffix_cells_moved": int(len(primary_suffix)),
                "new_track_suffix_cells_moved": int(len(candidate_suffix)),
            }
        )

    return {
        "examined_boundaries": int(examined_boundaries),
        "identity_switches_repaired": int(len(repairs)),
        "repairs": repairs,
        "parameters": {
            "max_frame_gap": int(max_frame_gap),
            "minimum_candidate_score": float(minimum_candidate_score),
            "minimum_improvement": float(minimum_improvement),
            "maximum_current_score": float(maximum_current_score),
        },
    }


def restore_untracked_cells(
    sctc: SingleCellTrajectoryCollection,
    all_segmented_cells: Iterable[SingleCellStatic],
    duplicate_iou_threshold: float = 0.95,
    chain_iou_threshold: float = 0.3,
    chain_max_gap: int = 1,
    forward_search_window: int = 2,
    minimum_link_iou: float = 0.3,
    workers: Optional[int] = None,
) -> dict:
    """Restore segmentation objects omitted by SORT without duplicating masks."""
    workers = workers or min(16, os.cpu_count() or 1)
    initial_index = _time_index(sctc)
    tracked_cell_keys = {
        (int(timeframe), str(getattr(sc, "id", "")))
        for timeframe, candidates in initial_index.items()
        for _, sc in candidates
    }
    tracked_label_keys = {
        (int(timeframe), int(sc.meta["label_in_mask"]))
        for timeframe, candidates in initial_index.items()
        for _, sc in candidates
        if sc.meta is not None and sc.meta.get("label_in_mask") is not None
    }


    def detect(sc):
        key = (int(sc.timeframe), str(getattr(sc, "id", "")))
        if key in tracked_cell_keys:
            return None
        label = sc.meta.get("label_in_mask") if sc.meta is not None else None
        if label is not None:
            label_key = (int(sc.timeframe), int(label))
            if label_key in tracked_label_keys:
                return None
        match = _best_iou(sc, initial_index.get(int(sc.timeframe), []), min_iou=-1.0)
        if match is not None and match[0] >= duplicate_iou_threshold:
            return None
        return sc

    cells = list(all_segmented_cells)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        missing = [sc for sc in executor.map(detect, cells) if sc is not None]

    chains: List[List[SingleCellStatic]] = []
    for sc in sorted(missing, key=lambda x: (int(x.timeframe), str(getattr(x, "id", "")))):
        best_index = None
        best_iou = None
        for index, chain in enumerate(chains):
            previous = chain[-1]
            gap = int(sc.timeframe) - int(previous.timeframe)
            if gap <= 0 or gap > chain_max_gap:
                continue
            iou = float(previous.compute_iou(sc))
            if iou > chain_iou_threshold and (best_iou is None or iou > best_iou):
                best_index, best_iou = index, iou
        if best_index is None:
            chains.append([sc])
        else:
            chains[best_index].append(sc)

    linked = added = skipped = 0
    for chain in chains:
        current_index = _time_index(sctc)
        clean = []
        for sc in chain:
            match = _best_iou(sc, current_index.get(int(sc.timeframe), []), min_iou=-1.0)
            if match is not None and match[0] >= duplicate_iou_threshold:
                skipped += 1
            else:
                clean.append(sc)
        if not clean:
            continue

        end_sc = clean[-1]
        future_best = None
        for gap in range(1, int(forward_search_window) + 1):
            timeframe = int(end_sc.timeframe) + gap
            for tid, candidate in _time_index(sctc).get(timeframe, []):
                iou = float(end_sc.compute_iou(candidate))
                if iou >= float(minimum_link_iou) and (
                    future_best is None
                    or iou > future_best[0]
                    or (iou == future_best[0] and gap < future_best[2])
                ):
                    future_best = (iou, int(tid), gap)

        if future_best is None:
            new_tid = int(sctc.get_max_tid()) + 1
            sctc.add_trajectory(
                SingleCellTrajectory(
                    track_id=new_tid,
                    timeframe_to_single_cell={int(sc.timeframe): sc for sc in clean},
                )
            )
            added += len(clean)
            continue

        target_tid = future_best[1]
        target = sctc.get_trajectory(target_tid)
        if {int(sc.timeframe) for sc in clean}.intersection(target.timeframe_set):
            skipped += len(clean)
            continue
        fragment = SingleCellTrajectory(
            track_id=target_tid,
            timeframe_to_single_cell={int(sc.timeframe): sc for sc in clean},
        )
        merged = target.copy()
        merged.add_nonoverlapping_sct(fragment)
        sctc.pop_trajectory_by_id(target_tid)
        sctc.add_trajectory(merged)
        linked += len(clean)

    return {
        "segmented_cells": len(cells),
        "missing_cells_detected": len(missing),
        "missing_chains": len(chains),
        "cells_linked_to_future_trajectories": linked,
        "cells_added_as_new_trajectories": added,
        "cells_skipped_as_duplicates": skipped,
    }


def reconnect_short_trajectories(
    sctc: SingleCellTrajectoryCollection,
    trajectory_length_threshold: int = 20,
    duplicate_iou_threshold: float = 0.95,
    forward_search_window: int = 2,
    minimum_link_iou: float = 0.3,
    require_mutual_best: bool = True,
    workers: Optional[int] = None,
) -> dict:
    """Reconnect mutually best non-overlapping fragments conservatively."""
    del workers  # Retained for API compatibility; indexed matching is deterministic.

    def duplicate_track_ids() -> List[int]:
        time_index = _time_index(sctc)
        lengths = {int(tid): len(traj) for tid, traj in sctc}
        duplicates = []
        for tid, traj in sctc:
            tid = int(tid)
            if len(traj) >= int(trajectory_length_threshold):
                continue
            is_duplicate = True
            for timeframe, sc in traj:
                found = False
                for candidate_tid, candidate in time_index.get(int(timeframe), []):
                    candidate_tid = int(candidate_tid)
                    if candidate_tid == tid:
                        continue
                    if lengths[candidate_tid] < lengths[tid]:
                        continue
                    if lengths[candidate_tid] == lengths[tid] and candidate_tid > tid:
                        continue
                    if float(sc.compute_iou(candidate)) >= float(duplicate_iou_threshold):
                        found = True
                        break
                if not found:
                    is_duplicate = False
                    break
            if is_duplicate:
                duplicates.append(tid)
        return duplicates

    duplicate_ids = duplicate_track_ids()
    for tid in duplicate_ids:
        if tid in sctc:
            sctc.pop_trajectory_by_id(tid)

    links = []
    while True:
        trajectories = {int(tid): traj for tid, traj in sctc}
        starts: Dict[int, List[int]] = {}
        for tid, traj in trajectories.items():
            starts.setdefault(int(traj.get_timeframe_span()[0]), []).append(tid)

        edges = []
        for source_tid, source in trajectories.items():
            source_end = int(source.get_timeframe_span()[1])
            source_last = source.get_sorted_scs()[-1]
            for gap in range(1, int(forward_search_window) + 1):
                for target_tid in starts.get(source_end + gap, []):
                    if target_tid == source_tid:
                        continue
                    target = trajectories[target_tid]
                    if min(len(source), len(target)) >= int(trajectory_length_threshold):
                        continue
                    target_first = target.get_sorted_scs()[0]
                    iou = float(source_last.compute_iou(target_first))
                    if iou >= float(minimum_link_iou):
                        edges.append((iou, gap, source_tid, target_tid))

        if not edges:
            break

        source_best = {}
        target_best = {}
        for edge in edges:
            iou, gap, source_tid, target_tid = edge
            source_key = (iou, -gap, -target_tid)
            target_key = (iou, -gap, -source_tid)
            if source_tid not in source_best or source_key > source_best[source_tid][0]:
                source_best[source_tid] = (source_key, edge)
            if target_tid not in target_best or target_key > target_best[target_tid][0]:
                target_best[target_tid] = (target_key, edge)

        candidates = []
        for _, edge in source_best.values():
            iou, gap, source_tid, target_tid = edge
            if require_mutual_best and target_best[target_tid][1][2] != source_tid:
                continue
            candidates.append(edge)
        if not candidates:
            break

        candidates.sort(key=lambda edge: (-edge[0], edge[1], edge[2], edge[3]))
        iou, gap, source_tid, target_tid = candidates[0]
        if source_tid not in sctc or target_tid not in sctc:
            continue
        source = sctc.get_trajectory(source_tid)
        target = sctc.get_trajectory(target_tid)
        if source.get_timeframe_span()[1] >= target.get_timeframe_span()[0]:
            continue

        if len(source) >= len(target):
            keeper_tid, removed_tid = source_tid, target_tid
            keeper, fragment = source.copy(), target
        else:
            keeper_tid, removed_tid = target_tid, source_tid
            keeper, fragment = target.copy(), source
        keeper.track_id = keeper_tid
        keeper.add_nonoverlapping_sct(fragment)
        sctc.pop_trajectory_by_id(source_tid)
        sctc.pop_trajectory_by_id(target_tid)
        sctc.add_trajectory(keeper)
        links.append(
            {
                "source_track_id": int(source_tid),
                "target_track_id": int(target_tid),
                "keeper_track_id": int(keeper_tid),
                "removed_track_id": int(removed_tid),
                "gap": int(gap),
                "endpoint_iou": float(iou),
            }
        )

    return {
        "duplicate_trajectories_removed": int(len(duplicate_ids)),
        "forward_merges": int(len(links)),
        "backward_merges": 0,
        "total_merges": int(len(links)),
        "links": links,
        "parameters": {
            "trajectory_length_threshold": int(trajectory_length_threshold),
            "forward_search_window": int(forward_search_window),
            "minimum_link_iou": float(minimum_link_iou),
            "require_mutual_best": bool(require_mutual_best),
        },
    }

def apply_post_sort_patches(
    sctc: SingleCellTrajectoryCollection,
    all_segmented_cells: Iterable[SingleCellStatic],
) -> Tuple[SingleCellTrajectoryCollection, dict]:
    """Apply the two required patches in their canonical order."""
    missing_report = restore_untracked_cells(sctc, all_segmented_cells)
    short_report = reconnect_short_trajectories(sctc)
    return sctc, {
        "restore_untracked_cells": missing_report,
        "reconnect_short_trajectories": short_report,
        "final_trajectories": int(len(sctc)),
        "final_cells": int(len(sctc.get_all_scs())),
    }
