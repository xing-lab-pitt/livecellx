import argparse
import datetime

import cv2
import skimage
import livecellx
from livecellx.model_zoo.segmentation.csn_sc_utils import correct_sc
from livecellx.model_zoo.segmentation.custom_transforms import CustomTransformEdtV9
from livecellx.model_zoo.segmentation.eval_csn import compute_watershed
from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux
from livecellx.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset
from livecellx.preprocess.utils import overlay

# %%
import numpy as np
import json
from livecellx.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from livecellx.core.single_cell import get_time2scs, show_sct_on_grid
from livecellx.core.datasets import LiveCellImageDataset
from livecellx.preprocess.utils import (
    overlay,
    enhance_contrast,
    normalize_img_to_uint8,
)
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.core.sc_filters import filter_boundary_cells, filter_scs_by_size

import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from typing import List

from typing import Dict
import tqdm
import livecellx.core.sc_mapping
from livecellx.segment.ou_simulator import find_contours_opencv, find_label_mask_contours


parser = argparse.ArgumentParser()
parser.add_argument("--DEBUG", action="store_true", help="Debug mode", default=False)
parser.add_argument("--min_area", type=int, help="Minimum area for a single cell", default=500)
args = parser.parse_args()

MIN_AREA = args.min_area


def get_zeromaps(sci2sci2metric, map_threshold=0.6):
    mapped_sci = {}
    res_zero_maps = {}
    for sci_1, sci2metric in sci2sci2metric.items():
        mapped_sci[sci_1] = []
        for sci_2, metric in sci2metric.items():
            if metric > map_threshold:
                if sci_2 not in mapped_sci:
                    mapped_sci[sci_2] = []
                mapped_sci[sci_1].append(sci_2)
        if len(mapped_sci[sci_1]) == 0:
            res_zero_maps[sci_1] = sci2metric
    return res_zero_maps


# def viz_csn_results(orig_sc, csn_scs, padding, out_dir, prefix=""):
#     nc = len(csn_scs) + 1
#     fig, axes = plt.subplots(2, nc, figsize=(nc * 5, 5), dpi=300)
#     if nc == 1:
#         axes = axes.reshape(2, 1)
#     axes[0, 0].imshow(orig_sc.get_img_crop(padding=padding))
#     axes[0, 0].set_title(f"Ref sc")
#     axes[1, 0].imshow(orig_sc.get_sc_mask(padding=padding))
#     axes[1, 0].set_title(f"Ref mask")
#     for i, _sc in enumerate(csn_scs):
#         axes[0, i + 1].imshow(_sc.get_img_crop(padding=padding))
#         axes[1, i + 1].imshow(_sc.get_sc_mask(padding=padding))
#     plt.savefig(out_dir / f"{prefix}corrected_sc_{orig_sc.id}_t{time}.png")
#     plt.close()


def viz_csn_mask_results(orig_sc: SingleCellStatic, out_mask, watershed_mask, padding, out_dir, prefix=""):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), dpi=300)
    axes[0].imshow(orig_sc.get_img_crop(padding=padding))
    axes[0].set_title(f"Ref sc")
    axes[1].imshow(orig_sc.get_sc_mask(padding=padding))
    axes[1].set_title(f"Ref mask")
    axes[2].imshow(out_mask.squeeze()[0])
    axes[2].set_title(f"CSN mask")
    axes[3].imshow(watershed_mask)
    axes[3].set_title(f"Watershed mask")

    plt.savefig(out_dir / f"{prefix}mask_corrected_sc_{orig_sc.id}_t{time}.png")
    plt.close()


def extend_and_fix_missing_sc(
    start_sc: SingleCellStatic,
    scs_by_time: Dict[int, List[SingleCellStatic]],
    model,
    threshold=0.7,
    metric="iou",
    padding=20,
    min_area=MIN_AREA,
    max_extend_time=10,
):
    # Pre-check if start-sc is valid or not
    res_dict: dict = correct_sc(start_sc, model, padding=padding, min_area=min_area, return_outputs=True)
    start_corrected_scs = res_dict["scs"]
    watershed_mask = res_dict["watershed_mask"]
    out_mask = res_dict["out_mask"]
    viz_csn_mask_results(
        start_sc,
        out_mask=out_mask,
        watershed_mask=watershed_mask,
        padding=padding,
        out_dir=csn_fig_dir,
        prefix="start-",
    )

    if len(start_corrected_scs) == 0:
        viz_csn_mask_results(
            start_sc,
            out_mask=out_mask,
            watershed_mask=watershed_mask,
            padding=padding,
            out_dir=csn_fig_dir,
            prefix="start-lack_corrected-",
        )
        return {
            "sct": SingleCellTrajectory(timeframe_to_single_cell={start_sc.timeframe: start_sc}),
            "state": "lack_correct_start",
        }
    elif len(start_corrected_scs) > 1:
        viz_csn_mask_results(
            start_sc,
            out_mask=out_mask,
            watershed_mask=watershed_mask,
            padding=padding,
            out_dir=csn_fig_dir,
            prefix="start-useg-",
        )
        return {
            "sct": SingleCellTrajectory(timeframe_to_single_cell={start_sc.timeframe: start_sc}),
            "state": "useg_start",
        }
    start_sc = start_corrected_scs[0]
    cur_time = start_sc.timeframe
    max_time = max([time for time in scs_by_time])
    res_scs = [start_sc]
    ref_sc = start_sc
    res_state = "complete"
    res_dict = {}
    for time in range(cur_time + 1, max_time + 1):
        if time - cur_time > max_extend_time:
            res_state = "end_by_max_extend_time"
            break
        if time not in scs_by_time:
            continue
        scs = scs_by_time[time]
        has_mapping = False
        for sc_next in scs:
            if metric == "iomin":
                computed_metric = ref_sc.compute_iomin(sc_next)
            elif metric == "iou":
                computed_metric = ref_sc.compute_iou(sc_next)
            else:
                raise ValueError("Invalid metric")

            if computed_metric > threshold:
                has_mapping = True
                break

        if has_mapping:
            break
        else:
            new_sc = SingleCellStatic(
                timeframe=time,
                img_dataset=ref_sc.img_dataset,
                mask_dataset=ref_sc.mask_dataset,
                contour=ref_sc.contour,
            )
            _scs = correct_sc(new_sc, model, padding=padding, min_area=MIN_AREA, input_transforms=input_transforms)
            assert isinstance(_scs, list), f"Invalid _scs type from correct_sc: {type(_scs)}"

            # Visualize correction results
            if len(_scs) == 0:
                res_state = "end_lack_corrected_scs"
                break
            # TODO: handle more than one sc case in _scs
            if len(_scs) > 1:
                res_state = "multiple_corrected_scs"
                viz_csn_mask_results(
                    start_sc,
                    out_mask=out_mask,
                    watershed_mask=watershed_mask,
                    padding=padding,
                    out_dir=csn_fig_dir,
                    prefix="loop-useg-multi-",
                )
                break

            # Visualize for debugging
            viz_csn_mask_results(
                start_sc,
                out_mask=out_mask,
                watershed_mask=watershed_mask,
                padding=padding,
                out_dir=csn_fig_dir,
                prefix="loop-corrected-",
            )

            res_scs.extend(_scs)
            ref_sc = _scs[0]
            res_state = "multi-complete"
    res_sct = SingleCellTrajectory(timeframe_to_single_cell={sc.timeframe: sc for sc in res_scs})
    res_dict = {
        "sct": res_sct,
        "state": res_state,
    }
    return res_dict


# def process_by_time(scs):
#     for sc in scs:
#         extended_sct = extend_and_fix_missing_sc(sc, cp_scs_by_time, model=model, threshold=0.5)
#         extended_filled_sctc.add_trajectory(extended_sct)


lcx_out_dir = Path("./notebook_results/CXA_process2_7_19/")
lcx_out_dir.mkdir(parents=True, exist_ok=True)

out_dir = Path("./notebook_results/CXA_process2_7_19/csn_fill_missing_cells/") / (
    "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
)
out_dir.mkdir(parents=True, exist_ok=True)
mask_dir = out_dir / "mask"
mask_dir.mkdir(parents=True, exist_ok=True)
sct_fig_dir = out_dir / "fig"
sct_fig_dir.mkdir(parents=True, exist_ok=True)
csn_fig_dir = out_dir / "csn_fig"
csn_fig_dir.mkdir(parents=True, exist_ok=True)

H_THRESHOLD = 1.3

# mapping_path = lcx_out_dir / "iomin_all_sci2sci2metric.json"
mapping_path = lcx_out_dir / "iou_all_sci2sci2metric.json"
sci2sci2metric = json.load(open(mapping_path, "r"))

# Loading model
model_ckpt = "/home/ken67/livecellx/notebooks/lightning_logs/version_v18_02-inEDTv1-augEdtV9-scaleV2-lr-0.0001-aux-seed-404/checkpoints/last.ckpt"
model = CorrectSegNetAux.load_from_checkpoint(model_ckpt)
input_transforms = CustomTransformEdtV9(degrees=0, shear=0, flip_p=0, use_gaussian_blur=True, gaussian_blur_sigma=30)
model.eval()
model.cuda()


if args.DEBUG:
    sctc_path = lcx_out_dir / "sctc_20-50.json"
    print("DEBUG MODE: Using sctc_20-50.json")
    cp_scs = SingleCellTrajectoryCollection.load_from_json_file(sctc_path).get_all_scs()
else:
    scs_path = lcx_out_dir / "single_cells_zero_based_time.json"
    cp_scs = SingleCellStatic.load_single_cells_json(scs_path)


# cp_scs = SingleCellTrajectoryCollection.load_from_json_file(
#     lcx_out_dir / "sctc_20-50.json"
# ).get_all_scs()

cp_scs_by_time = get_time2scs(cp_scs)
img_dataset = cp_scs[0].img_dataset
d2_mask_dataset = cp_scs[0].mask_dataset


id2sc = {sc.id: sc for sc in cp_scs}
filtered_scs = filter_scs_by_size(cp_scs, min_size=500)
filtered_scs = filter_boundary_cells(filtered_scs, 20)
valid_sc_ids = set([sc.id for sc in filtered_scs])

sci2sci2metric_keys = list(sci2sci2metric.keys())
filtered_sci2sci2metric = {}
for sci in sci2sci2metric_keys:
    if sci not in valid_sc_ids:
        continue
    filtered_sci2sci2metric[sci] = {}
    for sci2 in sci2sci2metric[sci]:
        if sci2 not in valid_sc_ids:
            continue
        filtered_sci2sci2metric[sci][sci2] = sci2sci2metric[sci][sci2]

sci2sci2metric = filtered_sci2sci2metric

zero_maps = get_zeromaps(sci2sci2metric, 0.5)
zero_sc_ids = list(zero_maps.keys())
zero_scs = [id2sc[sc_id] for sc_id in zero_sc_ids]
print("# zero_scs: ", len(zero_scs), "| # total_scs: ", len(cp_scs))

# Construct missing cells based on zero maps
td1_scs = []
for sc in zero_scs:
    sc_t1 = SingleCellStatic(
        timeframe=sc.timeframe + 1,
        img_dataset=sc.img_dataset,
        mask_dataset=sc.mask_dataset,
        contour=sc.contour,
    )
    sc_t1.meta["is_missing"] = True
    sc_t1.meta["from_sc"] = sc.id
    td1_scs.append(sc_t1)

td1_scs_by_time = get_time2scs(td1_scs)
print("# Area td1_scs: ", len(td1_scs))
filtered_td1_scs = filter_scs_by_size(td1_scs, 100)
print("# Area filtered_td1_scs: ", len(filtered_td1_scs))

extended_filled_sctc = SingleCellTrajectoryCollection()
extended_res_dicts = []

extend_df_dict = {
    "# extended": [],
    "state": [],
    "sc_id": [],
}

for time in tqdm.tqdm(td1_scs_by_time):
    print("current time: ", time, "| # scs: ", len(td1_scs_by_time[time]))
    for sc in td1_scs_by_time[time]:
        extended_dict = extend_and_fix_missing_sc(sc, cp_scs_by_time, model=model, threshold=0.7)
        extended_sct = extended_dict["sct"]
        extended_sct.meta["csn_corrected"] = True
        extended_state = extended_dict["state"]

        # Visualize the correct sct
        # nc = len(extended_sct)
        # show_sct_on_grid(extended_sct, interval=1, nc=nc, nr=1)
        # if len(extended_sct) > 1:
        #     plt.savefig(sct_fig_dir / f"multi-extended_sct_{sc.id}.png")
        # else:
        #     plt.savefig(sct_fig_dir / f"extended_sct_{sc.id}.png")
        # plt.close()

        extended_filled_sctc.add_trajectory(extended_sct)
        extended_res_dicts.append(extended_dict)

        # Save stats df
        extend_df_dict["# extended"].append(len(extended_sct))
        extend_df_dict["sc_id"].append(sc.id)
        extend_df_dict["state"].append(extended_state)
        tmp_df = pd.DataFrame(extend_df_dict)
        tmp_df.to_csv(out_dir / "extend_stats.csv", index=False)
        del tmp_df


extended_filled_sctc.write_json(out_dir / "extended_filled_sctc.json")

extended_states = [res_dict["state"] for res_dict in extended_res_dicts]
# Draw a histogram of the states, publication-ready

plt.figure(figsize=(10, 6))
plt.hist(extended_states, bins=30, edgecolor="black")
plt.title("Histogram of Extended States")
plt.xlabel("State")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "extended_states_hist.png", dpi=300)
plt.close()
