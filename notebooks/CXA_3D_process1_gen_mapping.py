# %%
import livecellx
from livecellx.preprocess.utils import overlay

# %%
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
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from typing import List

# %%
import glob
from PIL import Image, ImageSequence
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import tqdm
import os
import matplotlib.pyplot as plt
from livecellx.core.parallel import parallelize

DEBUG = False


img_dir = "/data/Ke_data/datasets/process1_30den/stitched/"
d2_img_dir = "/data/Ke_data/datasets/process1_30den/2D/"
d2_mask_dir = "/data/Ke_data/datasets/process1_30den/2D_mask/"
lcx_out_dir = Path("./tmp/CXA_process1/")
map_out_dir = lcx_out_dir
zero_map_dir = map_out_dir / "zero_maps_viz"
multi_map_dir = map_out_dir / "multi_map_viz"

os.makedirs(map_out_dir, exist_ok=True)
os.makedirs(zero_map_dir, exist_ok=True)
os.makedirs(multi_map_dir, exist_ok=True)


all_scs = SingleCellStatic.load_single_cells_json(lcx_out_dir / "cp_single_cells.json")
all_scs_by_time = get_time2scs(all_scs)

# For debugging
if DEBUG:
    all_scs = all_scs_by_time[0] + all_scs_by_time[1]

for _sc in tqdm.tqdm(all_scs, desc="Subsampling contours"):
    _sc.uns["full_contour"] = _sc.contour
    _sc.sample_contour_point(15)


# %%
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
    metric_key="iomin",
    min_map_num=None,
    metric="iomin",
):
    if metric == "iou":
        compute_scs_iou(scs_from, scs_to, metric_key)
    elif metric == "iomin":
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
    dt=1,
    total_time=None,
    interval=1,
    zero_map_dir=None,
    multi_map_dir=None,
    cores=16,
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
                "metric_key": "iomin",
                "multimap_metric_threshold": 0.2,
                "zeromap_metric_threshold": 0.8,
                "save_viz_check": True,
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


scs_by_time = get_time2scs(all_scs)

outputs = process_dt_scs_wrapper(
    scs_by_time,
    dt=1,
    total_time=None,
    interval=1,
    zero_map_dir=zero_map_dir,
    multi_map_dir=multi_map_dir,
    cores=32,
)

# %%
time2multi_maps__id = {}

new_scs = []
for output in outputs:
    time, multi_maps, zero_maps, scs_1 = output
    new_scs.extend(scs_1)
    time2multi_maps__id[time] = []
    for _sc, _mapped_scs in multi_maps.items():
        time2multi_maps__id[time].append({"map_from": str(_sc.id), "map_to": [str(sc.id) for sc in _mapped_scs]})


json.dump(time2multi_maps__id, open(map_out_dir / "time2multi_maps__id.json", "w"), indent=4)

# Recover full contour
for sc in new_scs:
    sc.contour = sc.uns["full_contour"]


all_sci2sci2metric = {}
for sc in new_scs:
    all_sci2sci2metric[sc.id] = {}
    for sc2, metric in sc.tmp["iomin"].items():
        if metric > 1e-5:
            all_sci2sci2metric[sc.id][sc2.id] = metric

json.dump(
    all_sci2sci2metric,
    open(map_out_dir / "all_sci2sci2metric.json", "w"),
    indent=4,
)
all_scs_ids = set([sc.id for sc in all_scs])
all_sci2sc = {sc.id: sc for sc in all_scs}
new_scs_ids = set([sc.id for sc in new_scs])
for sci in new_scs_ids:
    assert sci in all_scs_ids, f"sci {sci} not in all_scs_ids"

# Add original scs to new scs if id not in new scs
for sci in all_sci2sc:
    if sci not in new_scs_ids:
        new_scs.append(all_sci2sc[sci])
        new_scs_ids.add(sci)

assert len(new_scs) == len(all_scs), "New scs should have the same length as the original scs, all: {}, new: {}".format(
    len(all_scs), len(new_scs)
)

SingleCellStatic.write_single_cells_json(new_scs, lcx_out_dir / "cp_mapped_single_cells.json")
