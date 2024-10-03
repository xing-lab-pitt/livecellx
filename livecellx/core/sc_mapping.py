import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List

import glob
from PIL import Image, ImageSequence
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import tqdm
import os
import matplotlib.pyplot as plt
from livecellx.core.parallel import parallelize
import livecellx
from livecellx.preprocess.utils import overlay

import numpy as np
import json
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


def compute_scs_iou(scs_from: List[SingleCellStatic], scs_to: List[SingleCellStatic], key="iou"):
    for sc1 in scs_from:
        if key not in sc1.tmp:
            sc1.tmp[key] = {}
        for sc2 in scs_to:
            if sc2 in sc1.tmp[key]:
                pass
            else:
                sc1.tmp[key][sc2] = sc1.compute_iou(sc2)


def compute_scs_iomin(scs_from: List[SingleCellStatic], scs_to: List[SingleCellStatic], key="iomin"):
    for sc1 in scs_from:
        if key not in sc1.tmp:
            sc1.tmp[key] = {}
        for sc2 in scs_to:
            if sc2 in sc1.tmp[key]:
                pass
            else:
                sc1.tmp[key][sc2] = sc1.compute_iomin(sc2)


def find_maps(
    scs_from: List[SingleCellStatic],
    scs_to: List[SingleCellStatic],
    metric_threshold=0.3,
    metric_key=None,
    min_map_num=None,
    metric="iomin",
):
    if metric == "iou":
        if metric_key is None:
            metric_key = "iou"
        compute_scs_iou(scs_from, scs_to, metric_key)
    elif metric == "iomin":
        if metric_key is None:
            metric_key = "iomin"
        compute_scs_iomin(scs_from, scs_to, metric_key)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    scs_map = {}
    for sc1 in scs_from:
        scs_map[sc1] = []
        for sc2 in scs_to:
            if sc1.tmp[metric_key][sc2] > metric_threshold:
                scs_map[sc1].append(sc2)
    # Filter by length of scs_map elements
    if min_map_num is not None:
        scs_map = {k: v for k, v in scs_map.items() if len(v) >= min_map_num}
    return scs_map


def process_mapping_by_time(
    time,
    scs_1,
    scs_2,
    zero_map_viz_dir=None,
    multi_map_viz_dir=None,
    metric_key="iomin",
    metric="iomin",
    multimap_metric_threshold=0.2,
    zeromap_metric_threshold=0.2,
    save_viz_check=False,
    padding=50,
):
    if len(scs_1) == 0 or len(scs_2) == 0:
        return time, {}, {}
    time_1 = scs_1[0].timeframe
    time_2 = scs_2[0].timeframe

    multi_candidate_map = find_maps(
        scs_1,
        scs_2,
        metric_threshold=multimap_metric_threshold,
        metric_key=metric_key,
        metric=metric,
        min_map_num=None,
    )
    zero_candidate_map = find_maps(
        scs_1,
        scs_2,
        metric_threshold=zeromap_metric_threshold,
        metric_key=metric_key,
        metric=metric,
        min_map_num=0,
    )

    # Handling zero maps
    zero_maps = {k: v for k, v in zero_candidate_map.items() if len(v) == 0}
    zero_map_rate = len(zero_maps) / len(scs_1)
    if save_viz_check and zero_maps:
        # Number of axes is len(zero_maps) + 1, always >= 2
        fig, axs = plt.subplots(1, len(zero_maps) * 2 + 1, figsize=(5 * (len(zero_maps) + 1), 5))
        for idx, sc in enumerate(zero_maps.keys()):
            axs[idx * 2].imshow(sc.get_contour_mask(padding=padding))
            axs[idx * 2].set_title(f"Time {sc.timeframe} - sc-{idx}")
            axs[idx * 2 + 1].imshow(scs_2[0].get_img_crop(padding=padding, bbox=sc.bbox))
            axs[idx * 2 + 1].set_title(f"Time {time_2} with the same bbox")
        assert len(zero_maps) > 0
        sc = list(zero_maps.keys())[0]
        raw_img = sc.get_img()
        axs[-1].imshow(raw_img)
        plt.savefig(os.path.join(str(zero_map_viz_dir), f"zero_map_{time}.png"))
        plt.close(fig)

    # Handling multi maps
    multi_maps = {k: v for k, v in multi_candidate_map.items() if len(v) > 1}
    multi_map_rate = len(multi_maps) / len(scs_1)
    if save_viz_check and multi_maps:
        for sc1, sc2s in multi_maps.items():
            fig, axs = plt.subplots(1, len(sc2s) + 3, figsize=(5 * (len(sc2s) + 2), 5))
            axs[0].imshow(sc1.get_contour_mask(padding=padding))
            axs[0].set_title(f"Time {sc1.timeframe} - sc1")
            for idx, sc2 in enumerate(sc2s):
                axs[idx + 1].imshow(sc2.get_contour_mask(padding=padding, bbox=sc1.bbox))
                axs[idx + 1].set_title(f"Time {sc2.timeframe} - sc2_{idx}")
            raw_img = sc1.get_img_crop(padding=padding)
            axs[-2].imshow(sc2s[0].get_img_crop(padding=padding, bbox=sc1.bbox))
            axs[-2].set_title(f"Time {time_2} with the same sc1 bbox")
            axs[-1].imshow(raw_img)
            axs[-1].set_title(f"Time {time_1} raw img")
            plt.savefig(os.path.join(str(multi_map_viz_dir), f"multi_map_{time}_{sc1.id}.png"))
            plt.close(fig)

    return time, multi_maps, zero_maps, scs_1


def process_dt_scs_wrapper(
    scs_by_time,
    metric="iomin",
    dt=1,
    total_time=None,
    interval=1,
    zero_map_dir=None,
    multi_map_dir=None,
    cores=16,
    save_viz_check=False,
):
    # Ensure output directories exist
    if zero_map_dir is not None:
        os.makedirs(zero_map_dir, exist_ok=True)
    if multi_map_dir is not None:
        os.makedirs(multi_map_dir, exist_ok=True)
    if total_time is None:
        total_time = max(scs_by_time.keys())

    start_t = min(scs_by_time.keys())
    inputs = []
    for time in range(start_t, total_time - dt + 1, interval):
        if (time + dt not in scs_by_time) or (time not in scs_by_time):
            continue
        scs_2 = scs_by_time[time + dt]
        scs_1 = scs_by_time[time]
        # futures.append(executor.submit(process_mapping_by_time, time, selected_crappy_scs, scs_at_time, zero_map_dir, multi_map_dir))
        # process_mapping_by_time(time, selected_crappy_scs, scs_at_time, zero_map_dir, multi_map_dir)
        # inputs.append((time, scs_1, scs_2, zero_map_dir, multi_map_dir))
        inputs.append(
            {
                "time": time,
                "scs_1": scs_1,
                "scs_2": scs_2,
                "zero_map_viz_dir": zero_map_dir,
                "multi_map_viz_dir": multi_map_dir,
                "multimap_metric_threshold": 0.2,
                "zeromap_metric_threshold": 0.8,
                "save_viz_check": save_viz_check,
                "metric": metric,
                "metric_key": metric,
            }
        )
    print("# inputs:", len(inputs))
    counter = 0
    for _input in inputs:
        scs1, scs2 = _input["scs_1"], _input["scs_2"]
        counter += len(scs1) * len(scs2)
    print("Max cell pair count for map computation:", counter)
    outputs = parallelize(process_mapping_by_time, inputs, cores=cores)  # Max num of cores
    return outputs


def extend_zero_map(sc: SingleCellStatic, scs_by_time: Dict[int, List[SingleCellStatic]], threshold):
    """
    Extends the zero mapped cell by creating new SingleCellStatic objects for consecutive timeframes until a mapping is found.

    Args:
        sc (SingleCellStatic): The initial SingleCellStatic object.
        scs_by_time (List[SingleCellStatic]): A list of SingleCellStatic objects grouped by timeframe.
        threshold: The threshold value used to determine if a mapping exists between two SingleCellStatic objects.

    Returns:
        List[SingleCellStatic]: A list of SingleCellStatic objects representing the extended zero map.
    """
    cur_time = sc.timeframe
    max_time = max([sc.timeframe for time in scs_by_time for sc in scs_by_time[time]])
    res_scs = [sc]
    for time in range(cur_time + 1, max_time + 1):
        if time not in scs_by_time:
            continue
        scs = scs_by_time[time]
        has_mapping = False
        for sc_t1 in scs:
            # [TODO]: discuss whether to use iou or iomin
            iomin = sc.compute_iomin(sc_t1)
            if iomin > threshold:
                has_mapping = True
                break
        if has_mapping:
            break
        else:
            sc_new = SingleCellStatic(
                timeframe=time,
                img_dataset=sc.img_dataset,
                mask_dataset=sc.mask_dataset,
                contour=sc.contour,
            )
            res_scs.append(sc_new)
    return res_scs
