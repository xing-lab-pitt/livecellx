import datetime

import cv2
import skimage
import livecellx
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
from livecellx.segment.ou_simulator import find_contours_opencv


def get_zeromaps(sci2sci2metric, zeromap_threshold=0.6):
    zeromap_threshold_filtered_mapping = {}
    zero_maps = {}
    for sci_1, sci2metric in sci2sci2metric.items():
        zeromap_threshold_filtered_mapping[sci_1] = []
        for sci_2, metric in sci2metric.items():
            if metric > zeromap_threshold:
                if sci_2 not in zeromap_threshold_filtered_mapping:
                    zeromap_threshold_filtered_mapping[sci_2] = []
                zeromap_threshold_filtered_mapping[sci_1].append(sci_2)

        if len(zeromap_threshold_filtered_mapping[sci_1]) == 0:
            zero_maps[sci_1] = sci2metric
    return zero_maps


def correct_sc_mask(_sc, model, padding, input_transforms=None, gpu=True, h_threshold=1):
    raw_data = CorrectSegNetDataset._prepare_sc_inference_data(
        _sc, padding_pixels=padding, bbox=None, normalize_crop=True
    )
    original_shape = raw_data["augmented_raw_img"].shape[-2:]
    sample = CorrectSegNetDataset.prepare_and_augment_data(
        raw_data,
        input_type=model.input_type,
        bg_val=0,
        normalize_uint8=model.normalize_uint8,
        exclude_raw_input_bg=model.exclude_raw_input_bg,
        force_no_edt_aug=False,
        apply_gt_seg_edt=model.apply_gt_seg_edt,
        transform=input_transforms,
    )
    if gpu:
        outputs = model(sample["input"].unsqueeze(0).cuda())
    else:
        outputs = model(sample["input"].unsqueeze(0))
    label_out = outputs[1].cpu().detach().numpy().squeeze()
    label_str = CorrectSegNetDataset.label_onehot_to_str(label_out)

    from torchvision import transforms

    back_transforms = transforms.Compose(
        [
            transforms.Resize(size=(original_shape[0], original_shape[1])),
        ]
    )
    out_mask_transformed = back_transforms(outputs[0]).cpu().detach().numpy().squeeze()
    watershed_mask = compute_watershed(out_mask_transformed[0], h_threshold=h_threshold)
    return out_mask_transformed, watershed_mask, label_str


def contours_to_scs(contours, ref_sc: SingleCellStatic, padding=None, min_area=None):
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


def correct_sc(_sc, model, padding, input_transforms=None, gpu=True, min_area=4000):
    out_mask, watershed_mask, label_str = correct_sc_mask(_sc, model, padding, input_transforms, gpu)
    contours = find_contours_opencv(watershed_mask)
    _scs = contours_to_scs(contours, ref_sc=_sc, padding=padding, min_area=min_area)
    for sc_ in _scs:
        sc_.meta["csn_gen"] = True
        sc_.meta["orig_sc_id"] = str(_sc.id)

    return _scs


def viz_csn_results(orig_sc, csn_scs, padding, out_dir, prefix=""):
    nc = len(csn_scs) + 1
    fig, axes = plt.subplots(2, nc, figsize=(nc * 5, 5), dpi=300)
    if nc == 1:
        axes = axes.reshape(2, 1)
    axes[0, 0].imshow(orig_sc.get_img_crop(padding=padding))
    axes[0, 0].set_title(f"Ref sc")
    axes[1, 0].imshow(orig_sc.get_sc_mask(padding=padding))
    axes[1, 0].set_title(f"Ref mask")
    for i, _sc in enumerate(csn_scs):
        axes[0, i + 1].imshow(_sc.get_img_crop(padding=padding))
        axes[1, i + 1].imshow(_sc.get_sc_mask(padding=padding))
    plt.savefig(out_dir / f"{prefix}corrected_sc_{orig_sc.id}_t{time}.png")
    plt.close()


def extend_and_fix_missing_sc(
    start_sc: SingleCellStatic,
    scs_by_time: Dict[int, List[SingleCellStatic]],
    model,
    threshold=0.7,
    metric="iou",
    padding=20,
    min_area=4000,
):
    # Pre-check if start-sc is valid or not
    start_corrected_scs = correct_sc(start_sc, model, padding=padding, min_area=min_area)
    viz_csn_results(start_sc, start_corrected_scs, padding, csn_fig_dir)
    if len(start_corrected_scs) == 0:
        viz_csn_results(start_sc, start_corrected_scs, padding, csn_fig_dir, prefix="lack_corrected-")
        return {
            "sct": SingleCellTrajectory(timeframe_to_single_cell={start_sc.timeframe: start_sc}),
            "state": "lack_correct_start",
        }
    elif len(start_corrected_scs) > 1:
        viz_csn_results(start_sc, start_corrected_scs, padding, csn_fig_dir, prefix="useg-start-")
        return {
            "sct": SingleCellTrajectory(timeframe_to_single_cell={start_sc.timeframe: start_sc}),
            "state": "useg_start",
        }
    start_sc = start_corrected_scs[0]
    cur_time = start_sc.timeframe
    max_time = max([sc.timeframe for time in scs_by_time for sc in scs_by_time[time]])
    res_scs = [start_sc]
    ref_sc = start_sc
    res_state = "complete"
    res_dict = {}
    for time in range(cur_time + 1, max_time + 1):
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
            _scs = correct_sc(new_sc, model, padding=padding, min_area=4000, input_transforms=input_transforms)
            # Visualize correction results
            if len(_scs) == 0:
                res_state = "end_lack_corrected_scs"
                break

            # Visualize for debugging
            viz_csn_results(new_sc, _scs, padding, csn_fig_dir)

            # TODO: handle more than one sc case in _scs
            if len(_scs) > 1:
                res_state = "multiple_corrected_scs"
                viz_csn_results(new_sc, _scs, padding, csn_fig_dir, prefix="useg-multi-")
                break
            res_scs.extend(_scs)
            ref_sc = _scs[0]
            res_state = "multi-complete"
    res_sct = SingleCellTrajectory(timeframe_to_single_cell={sc.timeframe: sc for sc in res_scs})
    res_dict = {
        "sct": res_sct,
        "state": res_state,
    }
    return res_dict


def process_by_time(scs):
    for sc in scs:
        extended_sct = extend_and_fix_missing_sc(sc, cp_scs_by_time, model=model, threshold=0.5)
        extended_filled_sctc.add_trajectory(extended_sct)


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


cp_scs = SingleCellStatic.load_single_cells_json(lcx_out_dir / "single_cells_zero_based_time.json")
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

zero_maps = get_zeromaps(sci2sci2metric, 0.65)
zero_sc_ids = list(zero_maps.keys())
zero_scs = [id2sc[sc_id] for sc_id in zero_sc_ids]
print("# zero_scs: ", len(zero_scs))

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
for time in tqdm.tqdm(td1_scs_by_time):
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
