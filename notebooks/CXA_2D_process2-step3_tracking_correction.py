import argparse
import numpy as np
import json

import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from typing import List
import glob
from PIL import Image, ImageSequence
from pathlib import Path
import json
import tqdm
from skimage.measure import regionprops
import datetime

from scipy.optimize import linear_sum_assignment
from typing import Union
import seaborn as sns


from livecellx.core.sc_filters import filter_boundary_traj
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.model_zoo.segmentation import csn_configs
from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux
from livecellx.segment.ou_utils import create_ou_input_from_sc
from livecellx.core.utils import label_mask_to_edt_mask
import livecellx.segment.ou_viz
from livecellx.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from livecellx.core.single_cell import get_time2scs
from livecellx.core.datasets import LiveCellImageDataset
from livecellx.preprocess.utils import (
    overlay,
    enhance_contrast,
    normalize_img_to_uint8,
)
import livecellx
from livecellx.preprocess.utils import overlay
from livecellx.core.sc_filters import filter_boundary_cells
from livecellx.core.single_cell import compute_bbox_iomin, compute_bbox_overlap

# Add argparse
parser = argparse.ArgumentParser(description="Correct under-segmented cells")
parser.add_argument(
    "--DEBUG",
    action="store_true",
    help="If set, will only run on the first 10 time points",
    default=False,
)
args = parser.parse_args()

if args.DEBUG:
    print("Running in DEBUG mode!")


match_threshold = 0.5
match_search_interval = 3
dist_to_boundary = 50
h_threshold = 0.3

lcx_out_dir = Path("./notebook_results/CXA_process2_7_19/")
lcx_out_dir.mkdir(parents=True, exist_ok=True)
# SingleCellStatic.write_single_cells_json(all_scs, lcx_out_dir / "single_cells.json")
# mapping_path = lcx_out_dir / "all_sci2sci2metric.json"


# model_short_name = "v15"
# model_ckpt = "./lightning_logs/version_v15_02-augEdtV8-inEDT-scaleV2-lr=0.00001-aux/checkpoints/last.ckpt"
model_short_name = "v17"
model_ckpt = (
    "lightning_logs/version_v17_02-inEDTv1-augEdtV8-scaleV2-lr=0.00001-aux/checkpoints/epoch=76-test_loss=1.5893.ckpt"
)
model = CorrectSegNetAux.load_from_checkpoint(model_ckpt)
input_transforms = csn_configs.gen_train_transform_edt_v8(degrees=0, shear=0, flip_p=0)
model.cuda()
model.eval()

if args.DEBUG:
    correction_out_dir = lcx_out_dir / ("traj_logics_DEBUG")
    import shutil

    shutil.rmtree(correction_out_dir, ignore_errors=True)
else:
    correction_out_dir = lcx_out_dir / (
        f"traj_logics_correction_12-4_model-{model_short_name}-iomin-{match_threshold}_search_interval-{match_search_interval}"
    )

correction_out_dir.mkdir(parents=True, exist_ok=True)

################################################
# Redict stdout and stderr to log file
################################################
import logging
import sys

# Configure logging
log_file = correction_out_dir / "log.out"
logging.basicConfig(
    level=logging.INFO,  # Set level to INFO to exclude DEBUG messages
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(log_file, "w"), logging.StreamHandler()],
)

# Create a custom logger for your specific print statements
custom_logger = logging.getLogger("CUSTOM_LOGGER")
custom_logger.setLevel(logging.INFO)

# Redirect only your specific print statements to the custom logger
class StreamToCustomLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


# Use the custom logger for your specific print statements
sys.stdout = StreamToCustomLogger(custom_logger, logging.INFO)
sys.stderr = StreamToCustomLogger(custom_logger, logging.ERROR)

################################################
# End of redirecting stdout and stderr to log file
################################################

################################################
# End of redicting stdout and stderr to log file
################################################


# Write hyperparameters
hyperparams = {
    "match_threshold": match_threshold,
    "match_search_interval": match_search_interval,
    "filter_dist_to_boundary": dist_to_boundary,
    "csn_model_ckpt": model_ckpt,
    "h_threshold": h_threshold,
    "model_input": model.input_type,
    "model_short_name": model_short_name,
    "model_ckpt": model_ckpt,
}

with open(correction_out_dir / "hyperparams.json", "w") as f:
    json.dump(hyperparams, f, indent=4)

# track_sctc = SingleCellTrajectoryCollection.load_from_json_file(lcx_out_dir / "sctc_with_feature_change.json")
# track_sctc = SingleCellTrajectoryCollection.load_from_json_file(lcx_out_dir / "debug_long_sctc_with_features.json")
# track_sctc = SingleCellTrajectoryCollection.load_from_json_file(
#     lcx_out_dir / "sctc_20-50.json"
# )
load_sctc_path = lcx_out_dir / "sctc_filled_SORT_bbox.json"
if args.DEBUG:
    load_sctc_path = lcx_out_dir / "sctc_20-50.json"

track_sctc = SingleCellTrajectoryCollection.load_from_json_file(load_sctc_path)


original_sctc = SingleCellTrajectoryCollection.load_from_json_file(load_sctc_path)


# Checking id intersection, after getting all scs from sctc. ids are stored at sc.id
len(set([sc.id for sc in track_sctc.get_all_scs()]).intersection(set([sc.id for sc in original_sctc.get_all_scs()])))


# Filter track_sctc according to min/max time
min_time, max_time = 0, None
if args.DEBUG:
    min_time, max_time = 20, 25
filtered_sctc = SingleCellTrajectoryCollection()

for tid, sct in track_sctc:
    subsct = sct.subsct(min_time, max_time)
    if len(subsct) > 0:
        filtered_sctc.add_trajectory(subsct)


print("# of trajectories before filtering:", len(track_sctc))
print("# of trajectories after filtering:", len(filtered_sctc))

print("# of cells before filtering:", len(track_sctc.get_all_scs()))
print("# of cells after filtering:", len(filtered_sctc.get_all_scs()))

track_sctc = filtered_sctc


filtered_boundary_sctc = filter_boundary_traj(track_sctc, dist=dist_to_boundary)
print("# of trajectories before filtering by boundary dist:", len(track_sctc))
print("# of trajectories after filtering by boundary dist:", len(filtered_boundary_sctc))

track_sctc = filtered_boundary_sctc

print("Checking if all timeframes are consistent with sc.timeframe and sct timeframe keys...")
for sct in track_sctc.get_all_trajectories():
    for time in sct.timeframe_to_single_cell:
        assert time == sct.timeframe_to_single_cell[time].timeframe
print("[PASSED] All timeframes are consistent with sc.timeframe and sct timeframe keys")


# all_scs = prep_scs_from_mask_dataset(d2_mask_dataset, d2_mask_dataset)
all_scs = track_sctc.get_all_scs()
# all_scs = SingleCellStatic.load_single_cells_json(lcx_out_dir / "sijie_labeled_gt.json")
# all_scs = SingleCellTrajectoryCollection.load_from_json_file(lcx_out_dir / "single_cells_zero_based_time.json").get_all_scs()
# all_scs = SingleCellTrajectoryCollection.load_from_json_file(lcx_out_dir / "sctc_edited_8-28-2024.json").get_all_scs()
# all_scs = SingleCellTrajectoryCollection.load_from_json_file(lcx_out_dir / "sctc_edited_8-29-2024.json").get_all_scs()

# filtered_sctc.write_json(lcx_out_dir / f"sctc_{min_time}-{max_time}.json")


# # Select 5 longest trajectories as debug set
# trajs = track_sctc.get_all_trajectories()
# trajs.sort(key=lambda x: len(x), reverse=True)
# trajs = trajs[:5]
# debug_long_sctc = SingleCellTrajectoryCollection(scts=trajs)
# SingleCellTrajectoryCollection.write_json(debug_long_sctc, lcx_out_dir / "debug_long_sctc_with_features.json")


# expert_filed_scs = SingleCellTrajectoryCollection.load_from_json_file(
#     lcx_out_dir / "Sijie-labeled-missing_cells-9-29.json"
# ).get_all_scs()


fig, ax = plt.subplots(1, 1, figsize=(60, 5), dpi=300)
track_sctc.histogram_traj_length(ax=ax)
# x-ticks rotation
ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="right")
plt.savefig(correction_out_dir / "original_hist_traj_length.png")


def get_td1_pred_sc_mask_path(sc: SingleCellStatic):
    return (
        Path(
            "/home/ken67/livecellx/notebooks/notebook_results/CXA_process2_7_19/v15-CSN-experiments/out_threshold-1-padding-20-h_threshold-0.5/mask"
        )
        / f"sc_{sc.id}_watershed.npy"
    )


def read_td1_pred_sc_seg_mask(sc: SingleCellStatic):
    path = get_td1_pred_sc_mask_path(sc)
    if not path.exists():
        return None
    return np.load(get_td1_pred_sc_mask_path(sc))[0]


predicted_scs = []
for sc in all_scs:
    sc_path = get_td1_pred_sc_mask_path(sc)
    if not sc_path.exists():
        continue
    predicted_scs.append(sc)

print("# trajectories", len(track_sctc.get_all_trajectories()))
print("# total scs", len(all_scs))
print("# predicted scs", len(predicted_scs))


predicted_scs = filter_boundary_cells(list(predicted_scs), dist_to_boundary=dist_to_boundary)
print("# predicted scs after filtering", len(predicted_scs))


predicted_scs = set(predicted_scs)


annotation_table = []

TD_1_GT_path = lcx_out_dir / "multimap_annotation-Ke-SJ-10-15.csv"


def save_annotation_csv(anntoation_tuples):
    df = pd.DataFrame(annotation_tuples, columns=["sc_id", "label"])
    # Add timeframe column
    df["timeframe"] = df["sc_id"].apply(lambda x: id2sc[x].timeframe)
    df.to_csv(TD_1_GT_path, index=False)


def read_annotation_csv(save_path):
    if save_path.exists():
        return pd.read_csv(save_path)
    assert False, "Annotation file not found!"


annotation_table = read_annotation_csv(TD_1_GT_path)


annotation_table["label"].unique


def in_annotation_US(sc: SingleCellStatic):
    return (
        str(sc.id) in annotation_table["sc_id"].values
        and annotation_table[annotation_table["sc_id"] == str(sc.id)]["label"].values[0] == "US"
    )


def in_annotation_US_or_UNKNOWN(sc: SingleCellStatic):
    return str(sc.id) in annotation_table["sc_id"].values and (
        (annotation_table[annotation_table["sc_id"] == str(sc.id)]["label"].values[0] == "US")
        or (annotation_table[annotation_table["sc_id"] == str(sc.id)]["label"].values[0] == "UNKNOWN")
    )


all_scs = track_sctc.get_all_scs()
multimap_scs_gt_US = [sc for sc in all_scs if in_annotation_US(sc)]
multimap_scs_gt_US_or_UNKNOWN = [sc for sc in all_scs if in_annotation_US_or_UNKNOWN(sc)]
print("# Multimap GT scs:", len(multimap_scs_gt_US))
print("# predicted scs:", len(predicted_scs))
print("# Multimap GT scs or UNKNOWN:", len(multimap_scs_gt_US_or_UNKNOWN))

predicted_scs = multimap_scs_gt_US


trajectories = track_sctc.get_all_trajectories()
# trajectories[0].show_on_grid(nr=3, nc=3, interval=5);


track_missing_rate = []
for sct in trajectories:
    timespan = sct.get_time_span_length()
    tracked_cells = len(sct.get_all_scs())
    # print(f"timespan: {timespan}, tracked_cells: {tracked_cells}")
    # print(f"# cells / # frames: {tracked_cells / timespan}")
    track_missing_rate.append(1 - tracked_cells / timespan)

# Plot histogram of missing rate
plt.hist(track_missing_rate, bins=50)
plt.xlabel("Missing rate")
plt.ylabel("Frequency")


def get_sc_before_time(sct, time: float):
    cur_time = time
    times: Union[List[int], List[float]] = sct.timeframe_to_single_cell.keys()
    # Get the closest time
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
    cur_time = time
    times: Union[List[int], List[float]] = sct.timeframe_to_single_cell.keys()
    # Get the closest time
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


def show_sc_timelapse(sc: SingleCellStatic, before=2, after=2, padding=50):
    times = sorted(sc.img_dataset.time2url.keys())
    idx = times.index(sc.timeframe)
    before_idx = max(0, idx - before)
    after_idx = min(len(times), idx + after + 1)
    bbox = [int(x) for x in sc.bbox]
    print(bbox)
    fig, axes = plt.subplots(1, after_idx - before_idx, figsize=(3 * (after_idx - before_idx), 5), dpi=300)
    for i, ax in enumerate(axes):
        ax.imshow(sc.img_dataset.get_img_by_time(times[before_idx + i])[bbox[0] : bbox[2], bbox[1] : bbox[3]])
        time = times[before_idx + i]
        if time == sc.timeframe:
            ax.set_title(f"t={time} (current)")
        else:
            ax.set_title(f"t={time}")


def show_tracks(sctc: SingleCellTrajectoryCollection, denoted_scs, figsize=(5, 20), y_interval=10):
    """Plot each trajectory as a line with scatter points as one row. X-axis is time frame, Y-axis is track_id"""
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    track_y = 0
    for tid, sct in sctc:
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
    # Set y range, starting at 0
    ax.set_ylim(0, track_y)
    plt.show()
    return fig, ax


def show_tracks_missing(sctc: SingleCellTrajectoryCollection, denoted_pairs, figsize=(5, 40), y_interval=10):
    """
    Visualizes the missing tracks in a SingleCellTrajectoryCollection.

    Parameters
    ----------
    sctc : SingleCellTrajectoryCollection
        The collection of single cell trajectories.
    denoted_pairs : list of tuples
        A list of tuples where each tuple contains a pair of single cells (previous, next).
    figsize : tuple, optional
        The size of the figure to be created. Default is (5, 40).

    Returns
    -------
    tuple
        A tuple containing the figure and axis of the plot.

    The function plots the trajectories of single cells and highlights the denoted pairs with specific markers.
    Blue lines represent the trajectories, red 'x' markers indicate the under-segmented timeframes,
    and green 'o' markers indicate the previous single cells in the denoted pairs.
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    track_y = 0
    denoted_pairs_prev = [pair[0] for pair in denoted_pairs]
    denoted_to2prev = {pair[1]: pair[0] for pair in denoted_pairs}
    denoted_prev2times = {}
    denoted_prev2nexts = {}
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
        # Add sct track id text
        # ax.text(
        #     times[0],
        #     track_y,
        #     f"track_id: {tid}",
        #     fontsize=5,
        #     verticalalignment="bottom",
        # )
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
    # Set y range, starting at 0
    ax.set_ylim(0, track_y)
    plt.show()
    return fig, ax


###########################################################################
# CSN correction module
###########################################################################
def get_sc_watershed_path(sc: SingleCellStatic):
    # return Path("./notebook_results/CXA_process2_7_19/v15-CSN-experiments/out_threshold-1-padding-20-h_threshold-0.5/mask") / f"sc_{sc.id}_watershed.npy"
    # return (
    #     Path("./notebook_results/CXA_process2_7_19/traj_logics_correction_debug/mask")
    #     / f"sc_{sc.id}_watershed.npy"
    # )
    return correction_out_dir / "mask" / f"sc_{sc.id}_watershed.npy"


def read_sc_watershed_mask(sc: SingleCellStatic):
    path = get_sc_watershed_path(sc)
    if not path.exists():
        return None
    return np.load(get_sc_watershed_path(sc))


def is_sc_watershed_mask_exists(sc: SingleCellStatic):
    return get_sc_watershed_path(sc).exists()


def correct_missing_case(in_sc, missing_sc, out_dir: Path, padding=20, out_threshold=0.7, h_threshold=0.3):
    sct_in = in_sc.tmp["sct"]
    sct_missing = missing_sc.tmp["sct"]
    fig_dir = out_dir / "figs"
    mask_dir = out_dir / "mask"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    def correct_sc(_sc):
        # If alread exist, skip
        if is_sc_watershed_mask_exists(_sc):
            return

        ou_input = create_ou_input_from_sc(_sc, remove_bg=False, padding_pixels=padding)

        if ou_input is None:
            print("Skipping sc", _sc.id, "because its ou_input is None")
            return
        edt_mask = label_mask_to_edt_mask(ou_input > 0)

        # Retrieve raw image crop for EDT v1 case
        raw_img_crop = _sc.get_img_crop(padding=padding, preprocess_img_func=normalize_img_to_uint8).astype(float)
        raw_img_crop = normalize_img_to_uint8(raw_img_crop)

        seg_outputs, aux_output, watershed_mask = livecellx.segment.ou_viz.viz_ou_outputs(
            ou_input,
            ou_input > 0,
            model,
            input_transforms=input_transforms.apply_image_transforms,
            out_threshold=out_threshold,
            # show=True,
            show=False,
            save_path=fig_dir / f"sc_{_sc.id}.png",
            input_type=model.input_type,
            edt_mask=edt_mask,
            edt_transform=input_transforms.apply_mask_transforms,
            h_threshold=h_threshold,
            raw_crop=raw_img_crop,
        )
        seg_outputs = seg_outputs.detach().cpu().numpy()
        np.save(mask_dir / f"sc_{_sc.id}.npy", seg_outputs)
        np.save(mask_dir / f"sc_{_sc.id}_watershed.npy", watershed_mask)

    correct_sc(in_sc)
    correct_sc(missing_sc)


def update_sct(target_sct, new_sct, inplace=False):
    if not inplace:
        target_sct = target_sct.copy()
    for _time, _sc in new_sct:
        target_sct.timeframe_to_single_cell[_sc.timeframe] = _sc
    return target_sct


print(
    "Intersection of GT multimap cells and predicted cells:",
    len(set(predicted_scs).intersection(set(track_sctc.get_all_scs()))),
    len(predicted_scs),
)


def select_underseg_cells_by_missing(sctc, search_interval=2, threshold=0.5):
    potential_underseg_sc_pairs_by_missing_in_gt = []
    potential_underseg_sc_pairs_not_in_gt = []
    prev_or_after_missing_scs = set()
    underseg_candidate_pairs = []
    trajectories = sctc.get_all_trajectories()
    all_scs = sctc.get_all_scs()
    time2scs = get_time2scs(all_scs)
    # For debug
    prev2times = {}
    for sct in tqdm.tqdm(trajectories, desc="Selecting under-seg candidates by missing logic"):
        timespan = sct.get_time_span()
        head = timespan[0]
        end = timespan[1]

        for time in range(head, end + 1):
            if time not in sct.timeframe_to_single_cell:
                # Missing logic at time
                prev_sc = get_sc_before_time(sct, time)
                if prev_sc is None:
                    continue
                if int(prev_sc.timeframe) + search_interval < time:
                    continue
                if time not in time2scs:
                    continue

                prev_or_after_missing_scs.add(prev_sc)
                scs_at_time = time2scs[time]
                for cur_sc in scs_at_time:
                    # metric = compute_bbox_iomin(next_sc, prev_sc)
                    metric = cur_sc.compute_iomin(prev_sc)
                    # metric = compute_bbox_overlap(next_sc, prev_sc)
                    if metric > threshold:
                        underseg_candidate_pairs.append((prev_sc, cur_sc))
                        # print(f"{sct.track_id}: {prev_sc.id} -> {cur_sc.id} at time {time}, metric: {metric}")
                        # print(f"prev_sc.timeframe: {prev_sc.timeframe}, cur_sc.timeframe: {cur_sc.timeframe}")

        # Handle after missing logic
        for time in range(head, end + 1):
            if time not in sct.timeframe_to_single_cell:
                # Missing logic at time
                next_sc = get_sc_after_time(sct, time)
                if next_sc is None:
                    continue
                if int(next_sc.timeframe) - search_interval > time:
                    continue
                if time not in time2scs:
                    continue
                prev_or_after_missing_scs.add(next_sc)
                scs_at_time = time2scs[time]
                for cur_sc in scs_at_time:
                    # metric = compute_bbox_iomin(next_sc, prev_sc)
                    metric = cur_sc.compute_iomin(next_sc)
                    # metric = compute_bbox_overlap(next_sc, prev_sc)
                    if metric > threshold:
                        underseg_candidate_pairs.append((next_sc, cur_sc))
                        # print(f"{sct.track_id}: {prev_sc.id} -> {cur_sc.id} at time {time}, metric: {metric}")
                        # print(f"prev_sc.timeframe: {prev_sc.timeframe}, cur_sc.timeframe: {cur_sc.timeframe}")
    return {
        "prev_or_after_scs": prev_or_after_missing_scs,
        "underseg_candidate_pairs": underseg_candidate_pairs,
    }


def select_underseg_cells_by_end(sctc, search_interval=2, threshold=0.5):
    """For each trajecty end cell at time t, extend to t+search_interval to search for underseg candidates"""
    potential_underseg_sc_pairs_by_missing_in_gt = []
    potential_underseg_sc_pairs_not_in_gt = []
    prev_missing_scs = set()
    underseg_candidate_pairs = []
    end_cells = set()

    trajectories = sctc.get_all_trajectories()
    all_scs = sctc.get_all_scs()
    time2scs = get_time2scs(all_scs)

    # For debug
    for sct in tqdm.tqdm(trajectories):
        timespan = sct.get_time_span()
        head = timespan[0]
        end = timespan[1]
        if len(sct) == 0:
            continue
        end_cell = sorted(sct.get_all_scs(), key=lambda x: x.timeframe, reverse=True)[0]
        end_cells.add(end_cell)
        for time in range(end, end + search_interval + 1):
            if time not in time2scs:
                continue
            scs_at_time = time2scs[time]
            for cur_sc in scs_at_time:
                metric = cur_sc.compute_iomin(end_cell)
                if metric > threshold:
                    underseg_candidate_pairs.append((end_cell, cur_sc))
    return {
        "end_cells": end_cells,
        "underseg_candidate_pairs": underseg_candidate_pairs,
    }


def report_underseg_candidates(underseg_candidates, predicted_scs):
    underseg_candidates_not_in_gt = [sc for sc in underseg_candidates if sc not in predicted_scs]
    underseg_candidates_in_gt = [sc for sc in underseg_candidates if sc in predicted_scs]

    print("# of total cells:", len(all_scs))
    print("# of total GT multmaps cells:", len(predicted_scs))
    print(
        "Coverage of GT multimap cells:",
        len(underseg_candidates_in_gt) / len(predicted_scs),
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
    """
    Match two lists of single cells using the Linear Assignment Problem (LAP).

    Parameters:
    - list1: List of single cells (e.g., SingleCellStatic objects)
    - list2: List of single cells (e.g., SingleCellStatic objects)
    - cost_function: Function to compute the cost between two single cells

    Returns:
    - matches: List of tuples (index1, index2) representing the matched indices
    """
    # Create the cost matrix
    cost_matrix = np.zeros((len(scs_1), len(scs_2)))
    for i, sc1 in enumerate(scs_1):
        for j, sc2 in enumerate(scs_2):
            cost_matrix[i, j] = cost_function(sc1, sc2)

    # Solve the linear assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Construct the mapping from scs_1 to scs_2
    matches = [(i, j) for i, j in zip(row_ind, col_ind)]
    sc1_to_sc2 = {scs_1[i]: scs_2[j] for i, j in matches}
    return sc1_to_sc2


def fix_missing_trajectory(in_sc, missing_sc, padding=20, area_threshold=1000):
    sct_in_orig = in_sc.tmp["sct"]
    sct_missing_orig = missing_sc.tmp["sct"]

    sct_in = sct_in_orig.copy()
    sct_missing = sct_missing_orig.copy()

    if (not is_sc_watershed_mask_exists(missing_sc)) or (not is_sc_watershed_mask_exists(in_sc)):
        print("Missing mask for sc", in_sc.id, "or", missing_sc.id)
        print(get_sc_watershed_path(missing_sc))
        print(get_sc_watershed_path(in_sc))
        return {"case_type": "missing-mask", "state": "skipped"}

    # Load watershed mask
    watershed_mask_in = read_sc_watershed_mask(in_sc)
    watershed_mask_missing = read_sc_watershed_mask(missing_sc)

    assert watershed_mask_in is not None, "Watershed mask for in_sc is None"
    assert watershed_mask_missing is not None, "Watershed mask for missing_sc is None"

    in_region_props = regionprops(watershed_mask_in)
    missing_region_props = regionprops(watershed_mask_missing)

    # Filter out small regions
    in_region_props = [prop for prop in in_region_props if prop.area > area_threshold]
    missing_region_props = [prop for prop in missing_region_props if prop.area > area_threshold]

    def regionprops_to_scs(regionprops, orig_sc: SingleCellStatic, padding=padding):
        new_scs = []
        sc_bbox = orig_sc.get_bbox(padding=padding)
        for region in regionprops:
            coords = region.coords
            _contour = [(sc_bbox[0] + x, sc_bbox[1] + y) for x, y in coords]
            new_sc = SingleCellStatic(
                orig_sc.timeframe,
                contour=_contour,
                img_dataset=orig_sc.img_dataset,
            )
            new_scs.append(new_sc)
            new_sc.meta["orig_sc_id"] = str(orig_sc.id)
        return new_scs

    case_type = None
    res_dict = {}
    if len(in_region_props) == 1 and len(missing_region_props) > 1:
        case_type = "missing-US"
        new_scs = regionprops_to_scs(missing_region_props, missing_sc)
        res_dict["new_scs"] = new_scs
        if missing_sc.timeframe not in sct_missing.timeframe_to_single_cell:
            print(sct_missing.timeframe_to_single_cell.keys())
            print("missing time:", missing_sc.timeframe)
            print("is missing sc in sct_missing?", missing_sc in sct_missing.get_all_scs())

        # Now we have a list of new_scs, we need to match them to sct_missing and sct_in
        if missing_sc.timeframe in sct_missing.timeframe_to_single_cell:
            sct_missing.timeframe_to_single_cell.pop(missing_sc.timeframe)

        # Tricky here: we do not want to use missing_sc because it is potentially an underseg cell mask
        # Instead, we use the previous sc to match the new_scs
        temporal_neighbor_missing_sc = get_sc_before_time(sct_missing, missing_sc.timeframe)
        if temporal_neighbor_missing_sc is None:
            temporal_neighbor_missing_sc = get_sc_after_time(sct_missing, missing_sc.timeframe)

        is_lacking_neighbor = False
        if temporal_neighbor_missing_sc is None:
            is_lacking_neighbor = True
            _mapping = match_scs_by_lap(new_scs, [in_sc, missing_sc], cost_iou)
        else:
            _mapping = match_scs_by_lap(new_scs, [in_sc, temporal_neighbor_missing_sc], cost_iou)
        # Add new_scs to trajectories based on mapping
        mapped_orig_scs = set()
        for new_sc in new_scs:
            if new_sc not in _mapping:
                continue
            mapped_sc = _mapping[new_sc]
            mapped_orig_scs.add(mapped_sc)
            if mapped_sc == in_sc:
                sct_in.add_single_cell(new_sc)
            else:
                sct_missing.add_single_cell(new_sc)
        # Check if all the underseg pair scs are mapped
        if not is_lacking_neighbor:
            assert (
                len(mapped_orig_scs) == 2
            ), f"Not all the underseg pair scs are mapped:{mapped_orig_scs}, \n {_mapping}"
        else:
            assert (
                len(mapped_orig_scs) == 2
            ), f"Not all the underseg pair scs are mapped:{mapped_orig_scs}, \n {_mapping}"
        return {
            "case_type": case_type,
            "new_scs": new_scs,
            "updated_scts": [sct_in, sct_missing],
            "state": "fixed",
            "is_lacking_neighbor": is_lacking_neighbor,
        }
    elif len(missing_region_props) and len(in_region_props) > 1:
        case_type = "P/A-US"
        return {"case_type": case_type, "state": "skipped"}
    elif len(missing_region_props) == 1 and len(in_region_props) == 1:
        case_type = "Both-correct"
        return {"case_type": case_type, "state": "skipped"}
    elif len(missing_region_props) > 1 and len(in_region_props) > 1:
        case_type = "Both-US"
        return {"case_type": case_type, "state": "skipped"}
    else:
        case_type = "Others"
        return {"case_type": case_type, "state": "skipped"}


cur_sctc = track_sctc
visited_pairs = set()
for round in range(1, 50):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Round", round)
    print("# of visited pairs:", len(visited_pairs))
    print("# of cells in cur_sctc:", len(cur_sctc.get_all_scs()))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # Add "sct" tmp key to each sc, for mapping sc to sct
    for tid, sct in cur_sctc:
        for sc in sct.get_all_scs():
            sc.tmp["sct"] = sct

    round_out_dir = correction_out_dir / f"round_{round}"
    round_out_dir.mkdir(parents=True, exist_ok=True)

    res_dict = select_underseg_cells_by_missing(
        cur_sctc, threshold=match_threshold, search_interval=match_search_interval
    )
    prev_or_after_scs = res_dict["prev_or_after_scs"]
    underseg_candidate_pairs_by_missing = res_dict["underseg_candidate_pairs"]
    underseg_candidates_by_missing = set([pair[1] for pair in underseg_candidate_pairs_by_missing])

    print(
        "=====================================",
        "Missing logic",
        "=====================================",
    )
    report_underseg_candidates(underseg_candidates_by_missing, predicted_scs)
    print("Filtering out visited pairs...")
    num_before_filtering = len(underseg_candidate_pairs_by_missing)
    underseg_candidate_pairs_by_missing = [
        pair for pair in underseg_candidate_pairs_by_missing if pair not in visited_pairs
    ]
    print(
        "Number of pairs before Filtering out visited pairs:",
        num_before_filtering,
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

    report_underseg_candidates(underseg_candidates_by_missing, predicted_scs)
    print(
        "# of consecutive scs by missing logics that are not in predicted scs:",
        len(consecutive_underseg_scs_underseg_scs_not_in_gt),
    )
    print(
        "=====================================",
        "End",
        "=====================================",
    )

    print(
        "=====================================",
        "Ending Logics",
        "=====================================",
    )
    res_dict = select_underseg_cells_by_end(cur_sctc, threshold=match_threshold, search_interval=match_search_interval)
    end_scs = res_dict["end_cells"]
    underseg_candidate_pairs_by_ending = res_dict["underseg_candidate_pairs"]
    underseg_candidates_by_ending = set([pair[1] for pair in underseg_candidate_pairs_by_ending])

    report_underseg_candidates(underseg_candidates_by_ending, predicted_scs)

    print("Filtering out visited pairs...")
    num_before_filtering = len(underseg_candidate_pairs_by_ending)
    underseg_candidate_pairs_by_ending = [
        pair for pair in underseg_candidate_pairs_by_ending if pair not in visited_pairs
    ]
    print(
        "Number of pairs before Filtering out visited pairs:",
        num_before_filtering,
        "Number of pairs after Filtering out visited pairs:",
        len(underseg_candidate_pairs_by_ending),
    )
    visited_pairs = visited_pairs.union(set(underseg_candidate_pairs_by_ending))

    consecutive_underseg_scs_underseg_scs_not_in_gt = [
        _two_scs
        for _two_scs in underseg_candidate_pairs_by_ending
        if _two_scs[0].timeframe + 1 == _two_scs[1].timeframe
    ]
    print(
        "# of consecutive scs by missing logics that are not in predicted scs:",
        len(consecutive_underseg_scs_underseg_scs_not_in_gt),
    )
    print(
        "=====================================",
        "End",
        "=====================================",
    )

    print(
        "=====================================",
        "Missing + End Summary",
        "=====================================",
    )

    num_covered_by_logics = len(
        set(underseg_candidates_by_ending).union(set(underseg_candidates_by_missing)).intersection(predicted_scs)
    )

    print(
        "Percentage of TD-1 GT multimap cell found by logics",
        num_covered_by_logics / len(predicted_scs),
    )

    report_underseg_candidates(underseg_candidates_by_missing, multimap_scs_gt_US_or_UNKNOWN)
    print(
        "=====================================",
        "End",
        "=====================================",
    )

    show_tracks_missing(
        cur_sctc.filter_trajectories_by_length(min_length=30),
        underseg_candidate_pairs_by_missing,
        figsize=(36, (5.0 / 30) * len(cur_sctc)),
        y_interval=1,
    )
    plt.savefig(round_out_dir / "before_correct_missing_track_plot.png")

    underseg_pairs_all = set(underseg_candidate_pairs_by_missing).union(underseg_candidate_pairs_by_ending)
    underseg_pairs_all = list(underseg_pairs_all)

    if len(underseg_pairs_all) == 0:
        print("[INFO] No underseg pairs to be fixed found in this round, exiting...")
        break

    for underseg_candidate_pair in tqdm.tqdm(underseg_pairs_all, desc="Correcting masks"):
        correct_missing_case(
            underseg_candidate_pair[0],
            underseg_candidate_pair[1],
            out_dir=correction_out_dir,  # NOT round_out_dir
            padding=20,
            h_threshold=h_threshold,
        )

    fixed_attempt_counter = 0
    case_types = []
    fixed_result_dicts = []
    for underseg_candidate_pair in tqdm.tqdm(underseg_pairs_all, desc="Fixing by logics"):
        assert underseg_candidate_pair[0] in cur_sctc.get_all_scs(), "Missing sc: {}".format(
            underseg_candidate_pair[0].id
        )

        res = fix_missing_trajectory(underseg_candidate_pair[0], underseg_candidate_pair[1])
        fixed_attempt_counter += 1
        case_types.append(res["case_type"])
        fixed_result_dicts.append(res)

    # Plot case types distribution
    print("[Start] Distribution of case types")
    # plt.figure(figsize=(3, 4), dpi=300)
    fig, ax = plt.subplots(1, 1, figsize=(3, 4), dpi=300)
    sns.set_theme(style="whitegrid", palette="pastel")
    sns.histplot(
        case_types,
        kde=False,
        color="blue",
        edgecolor="black",
        ax=ax,
    )

    # Set title and labels with increased font sizes
    plt.title("Distribution of Case Types", fontsize=16, fontweight="bold")
    plt.xlabel("Case Type", fontsize=14, fontweight="bold")
    plt.ylabel("Count", fontsize=14, fontweight="bold")

    # Set tick parameters and rotate x-axis ticks
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)

    fig.tight_layout()
    plt.savefig(round_out_dir / "case_type_hist.png")
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
    for idx in range(len(underseg_pairs_all)):
        fix_res_dict = fixed_result_dicts[idx]
        underseg_candidate_pair = underseg_pairs_all[idx]
        if fix_res_dict["state"] == "skipped":
            continue
        elif fix_res_dict["state"] == "fixed":
            # TODO: pop_trajectory -> pop_trajectory_by_id
            fixed_case_counter += 1
            # Important: Do not use sct stored in tmp directly
            # Becuase it may not be the latest sct in cur_sctc
            track_id = underseg_candidate_pair[0].tmp["sct"].track_id
            traj_in_sctc = corrected_sctc[track_id]
            _updated_sct1 = update_sct(traj_in_sctc, fix_res_dict["updated_scts"][0])
            corrected_sctc.pop_trajectory_by_id(_updated_sct1.track_id)
            corrected_sctc.add_trajectory(_updated_sct1)
            underseg_candidate_pair[0].tmp["sct"] = _updated_sct1

            # Update the second sc
            if len(fix_res_dict["updated_scts"]) > 1:
                second_track_id = underseg_candidate_pair[1].tmp["sct"].track_id
                second_traj_in_sctc = corrected_sctc[second_track_id]
                _updated_sct_second = update_sct(
                    second_traj_in_sctc,
                    fix_res_dict["updated_scts"][1],
                )
                corrected_sctc.pop_trajectory_by_id(_updated_sct_second.track_id)
                corrected_sctc.add_trajectory(_updated_sct_second)
                underseg_candidate_pair[1].tmp["sct"] = _updated_sct_second

    print("*" * 100)
    print(f"* Round-{round} Summary")

    print("Corrected SCTC: {} trajectories, Original SCTC: {} trajectories".format(len(corrected_sctc), len(cur_sctc)))
    print("# of Total cells in corrected SCTC:", len(corrected_sctc.get_all_scs()))
    print("# of Total cells in round-start SCTC:", len(cur_sctc.get_all_scs()))
    print("# Fixed case:", fixed_case_counter)
    print(
        "Number of fix attempts by CSN in this round:",
        fixed_attempt_counter,
    )
    print(
        "Percentage of cases fixed by CSN in this round:",
        fixed_case_counter / fixed_attempt_counter,
    )

    print("*" * 100)
    cur_sctc = corrected_sctc

    fig, ax = plt.subplots(1, 1, figsize=(40, 5), dpi=300)
    filtered_original_sctc = filter_boundary_traj(original_sctc, dist=dist_to_boundary)
    filtered_original_sctc.histogram_traj_length(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.savefig(round_out_dir / "original_hist_traj_length.png")

    fig, ax = plt.subplots(1, 1, figsize=(40, 5), dpi=300)
    corrected_sctc.histogram_traj_length(ax=ax, color="blue")
    # Rotate x-axis ticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.savefig(round_out_dir / "corrected_hist_traj_length.png")

corrected_sctc.write_json(
    correction_out_dir / (f"corrected_sctc-final-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json")
)
