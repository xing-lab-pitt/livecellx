# %%
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.io import imread
import glob
from pathlib import Path
from PIL import Image, ImageSequence
from tqdm import tqdm
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from livecell_tracker import segment
from livecell_tracker import core
from livecell_tracker.core import datasets
from livecell_tracker.core.datasets import LiveCellImageDataset
from skimage import measure
from livecell_tracker.core import SingleCellTrajectory, SingleCellStatic

# import detectron2
# from detectron2.utils.logger import setup_logger
import tqdm

# setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import cv2

# import some common detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from livecell_tracker.segment.detectron_utils import gen_cfg

# from livecell_tracker.segment.detectron_utils import (
#     segment_detectron_wrapper,
#     segment_images_by_detectron,
#     convert_detectron_instance_pred_masks_to_binary_masks,
#     convert_detectron_instances_to_label_masks,
# )
# from livecell_tracker.segment.detectron_utils import (
#     convert_detectron_instance_pred_masks_to_binary_masks,
#     convert_detectron_instances_to_label_masks,
#     segment_images_by_detectron,
#     segment_single_img_by_detectron_wrapper,
# )
from livecell_tracker.annotation.coco_utils import coco_to_sc
from pycocotools.coco import COCO
from livecell_tracker.segment.utils import match_mask_labels_by_iou
from livecell_tracker.preprocess.utils import enhance_contrast, normalize_img_to_uint8


def compute_match_label_map(t1, t2, mask_dataset, iou_threshold=0.2):
    label_mask1 = mask_dataset.get_img_by_time(t1)
    label_mask2 = mask_dataset.get_img_by_time(t2)

    # Note: first arg is mask2 and second arg is mask1 to create mask1 label to mask2 label map
    # read match_mask_labels_by_iou docstring for more info
    _, score_dict = match_mask_labels_by_iou(label_mask2, label_mask1, return_all=True)
    label_map = {}
    for label_1 in score_dict:
        label_map[label_1] = {}
        for score_info in score_dict[label_1]:
            if score_info["iou"] > iou_threshold:
                label_map[label_1][score_info["seg_label"]] = {"iou": score_info["iou"]}
    return t1, t2, label_map


def set_sc_label(sc, bg_val=0):
    """Assume sc.mask_dataset contains label masks"""
    label_mask_pixels = sc.mask_dataset.get_img_by_time(sc.timeframe)[sc.bbox[0] : sc.bbox[2], sc.bbox[1] : sc.bbox[3]][
        sc.get_contour_mask()
    ]
    labels = np.unique(label_mask_pixels)
    labels = list(set(labels))
    if bg_val in labels:
        labels.remove(bg_val)  # remove bg label

    # TODO: figure out why skimage regionprops sometimes returns contours containing other labels...
    # it should be a bug in skimage
    if len(labels) != 1:
        label_info = []
        for i, label in enumerate(labels):
            label_sum = (label_mask_pixels == label).sum()
            label_info.append((label, label_sum))
        label_info = sorted(label_info, key=lambda x: x[1], reverse=True)
        # assert label_info[0][1] > (0.9 * sc.get_contour_mask().sum()), r"no label exceeds 90% of the sc contour mask, percent: {}".format(label_info[0][1] / sc.get_contour_mask().sum())
        if label_info[0][1] < (0.9 * sc.get_contour_mask().sum()):
            print(
                "Warning: no label exceeds 90% of the sc contour mask, percent: {}".format(
                    label_info[0][1] / sc.get_contour_mask().sum()
                )
            )
            print("sc time", sc.timeframe, "labels:", labels)
            print("label info (label, pixels):", label_info, "percent:", label_info[0][1] / sc.get_contour_mask().sum())
            # sc.show_panel()
            # plt.show()
        labels = [label_info[0][0]]
        # sc.show_panel()
    assert len(labels) == 1, labels
    sc.id = list(labels)[0]
    return sc


if __name__ == "__main__":
    # TODO: change paths according to your local setup
    pos_path = Path("XY16")

    dataset_dir_path = Path(
        "../datasets/EBSS_Starvation/tif_STAV-A549_VIM_24hours_NoTreat_NA_YL_Ti2e_2022-12-21" / pos_path
    )

    mask_dataset_path = Path(
        "../datasets/EBSS_Starvation/tif_STAV-A549_VIM_24hours_NoTreat_NA_YL_Ti2e_2022-12-21/out" / pos_path / "seg"
    )

    out_dir = Path("./notebook_results/train_real_td_data/")

    # %%
    def get_time_from_path(path):
        """example path: STAV-A549_VIM_24hours_NoTreat_NA_YL_Ti2e_2022-12-21_T287_XY09_TRITC.tif"""
        idx = 0
        strs = path.split("_")
        while idx < len(strs) - 1:
            if strs[idx][:2] == "XY":
                break
            idx += 1
        idx -= 1
        return int(strs[idx][1:])

    get_time_from_path("example path: STAV-A549_VIM_24hours_NoTreat_NA_YL_Ti2e_2022-12-21_T287_XY09_DIC.tif")

    # %%
    # MAX_IMG_NUM = 10
    MAX_IMG_NUM = int(1e20)

    # %% [markdown]
    # Sample time points by interval

    # %%
    mask_paths = sorted(glob.glob(str(mask_dataset_path / "*.png")))[:MAX_IMG_NUM]
    total_time = len(mask_paths)
    sample_num = 40
    # sample_num = 3
    sample_interval = total_time // sample_num
    times = np.linspace(1, total_time, sample_num)

    # note that we need image (t, t+1) to search for over/under-segmentation pairs
    times = set([int(t) for t in times] + [int(t + 1) for t in times])

    # %%
    mask_time2url = {}
    mask_paths = sorted(glob.glob(str(mask_dataset_path / "*.png")))[:MAX_IMG_NUM]
    for mask_path in mask_paths:
        print(mask_path)
        mask_time2url[get_time_from_path(mask_path)] = mask_path

    mask_time2url = {k: v for k, v in mask_time2url.items() if k in times}
    label_mask_dataset = LiveCellImageDataset(ext="png", time2url=mask_time2url)
    len(label_mask_dataset)

    # %%
    time2url = {}
    img_paths = sorted(glob.glob(str(dataset_dir_path / "*_DIC.tif")))[:MAX_IMG_NUM]

    for img_path in img_paths:
        print(img_path)
        time = get_time_from_path(img_path)
        time2url[time] = img_path

    time2url = {k: v for k, v in time2url.items() if k in times}
    dic_dataset = LiveCellImageDataset(dataset_dir_path, time2url=time2url, ext="tif")

    # %% [markdown]
    # check co-existence of times

    # %%
    for time in label_mask_dataset.time2url:
        assert time in dic_dataset.time2url

    for time in dic_dataset.time2url:
        assert time in label_mask_dataset.time2url

    print("<data loading done>")
    print("total time points:", len(label_mask_dataset))
    # %% [markdown]
    # Convert label masks to single cell objects

    # %%
    from multiprocessing import Pool
    from skimage.measure import regionprops, find_contours
    from livecell_tracker.segment.ou_simulator import find_contours_opencv

    from livecell_tracker.segment.utils import prep_scs_from_mask_dataset

    single_cells = prep_scs_from_mask_dataset(label_mask_dataset, dic_dataset, cores=None)
    # single_cells = single_cells[:5]

    from livecell_tracker.core.parallel import parallelize

    # %%

    single_cells = parallelize(set_sc_label, [[sc] for sc in single_cells], cores=None)

    # %%
    for sc in single_cells:
        assert sc

    # %%
    # st = SingleCellTrajectory(track_id=-1, timeframe_to_single_cell={idx:sc for idx, sc in enumerate(single_cells)}, img_dataset=dic_dataset, mask_dataset=label_mask_dataset)
    # st.timeframe_to_single_cell = {idx:sc for idx, sc in enumerate(single_cells)}
    # st.write_json("notebook_results/single_traj.json")

    # %%
    # for testing
    # single_cells = single_cells[:10]

    # %%
    single_cells_by_time = {}
    for cell in single_cells:
        if cell.timeframe not in single_cells_by_time:
            single_cells_by_time[cell.timeframe] = []
        single_cells_by_time[cell.timeframe].append(cell)

    # %%
    times = sorted(single_cells_by_time.keys())
    for time in times[:5]:
        print(time, len(single_cells_by_time[time]))

    # %% [markdown]
    # Visualize one single cell

    # %%
    sc = single_cells[0]

    times = sorted(label_mask_dataset.times)
    inputs = []
    for idx in times:
        t1 = idx
        if t1 + 1 in times:
            t2 = t1 + 1
        else:
            continue
        inputs.append((t1, t2, label_mask_dataset))
    label_match_outputs = parallelize(compute_match_label_map, inputs, None)

    # %%
    multiple_maps = []
    for t1, t2, label_map in label_match_outputs:
        for label in label_map:
            if len(label_map[label]) > 1:
                # print(t1, t2, label, label_map[label])
                multiple_maps.append((t1, t2, label, label_map[label]))

    # %%
    time2id2sc = {}
    for sc in single_cells:
        time = sc.timeframe
        if time not in time2id2sc:
            time2id2sc[time] = {}
        if sc.id in time2id2sc[time]:
            print("Warning: sc id already exists in time2id2sc, sc id: {}, time: {}".format(sc.id, time))
            # sc.show_panel()
            # time2id2sc[time][sc.id].show_panel()

        assert sc.id not in time2id2sc[time]
        time2id2sc[time][sc.id] = sc

    # %% [markdown]
    # Sort multiple_maps by t1

    # %%
    multiple_maps = sorted(multiple_maps, key=lambda x: x[0])

    # %%

    def human_loop_answer_over_under_seg(multiple_maps, time2id2sc, padding=80):
        over_maps = []
        under_maps = []
        discarded_maps = []
        fig_offset = 4

        for info in tqdm.tqdm(multiple_maps):
            t1, t2, label, mapping = info
            print("t1:", t1, "t2:", t2, "label:", label)
            print("info: ", info)
            t1_sc = time2id2sc[t1][label]
            t2_scs = []
            for tmp_label in mapping:
                sc = time2id2sc[t2][tmp_label]
                t2_scs.append(sc)
            # fig, axes = plt.subplots(1, len(t2_scs) * 2 + offset, figsize=(60, 30))
            fig, axes = plt.subplots(1, fig_offset, figsize=(60, 30))
            axes[0].imshow(enhance_contrast(normalize_img_to_uint8(t1_sc.get_img()), factor=10))
            axes[0].set_title("t1 img")
            axes[1].imshow(t1_sc.get_mask_crop(padding=padding, bbox=t1_sc.bbox, dtype=int))
            axes[1].set_title("t1 label mask")
            # t2_scs[0].show_mask(ax=axes[2], padding=padding, crop=True, bbox=t1_sc.bbox)
            axes[2].imshow(t2_scs[0].get_mask_crop(padding=padding, bbox=t1_sc.bbox, dtype=int))
            axes[2].set_title("t2 label mask")

            # t1_sc.show(crop=True, ax=axes[3], padding=padding)
            axes[3].imshow(enhance_contrast(normalize_img_to_uint8(t1_sc.get_img_crop(padding=padding))))
            axes[3].set_title("t1 img crop")

            def sc_rect(sc, relative_bbox=None, padding=0, color="r", on_crop=False):
                if on_crop:
                    bbox = sc.get_bbox_on_crop(padding=padding, bbox=relative_bbox)
                else:
                    bbox = sc.bbox
                return patches.Rectangle(
                    (bbox[1], bbox[0]),
                    (bbox[3] - bbox[1]),
                    (bbox[2] - bbox[0]),
                    linewidth=1,
                    edgecolor=color,
                    facecolor="none",
                )

            axes[0].add_patch(sc_rect(t1_sc, color="b"))
            axes[1].add_patch(sc_rect(t1_sc, relative_bbox=t1_sc.bbox, padding=padding, color="b", on_crop=True))
            axes[2].add_patch(sc_rect(t1_sc, relative_bbox=t1_sc.bbox, padding=padding, color="b", on_crop=True))
            axes[3].add_patch(sc_rect(t1_sc, relative_bbox=t1_sc.bbox, padding=padding, color="b", on_crop=True))

            # # show individual sc in t2
            for idx, sc in enumerate(t2_scs):
                # sc.show_contour_mask(ax=axes[idx*2 + offset], padding=padding)
                # sc.show_mask(ax=axes[idx*2 + 1 + offset], padding=padding, crop=True)
                axes[0].add_patch(sc_rect(sc))
                axes[1].add_patch(sc_rect(sc, relative_bbox=t1_sc.bbox, padding=padding, color="r", on_crop=True))
                axes[2].add_patch(sc_rect(sc, relative_bbox=t1_sc.bbox, padding=padding, color="r", on_crop=True))
                axes[3].add_patch(sc_rect(sc, relative_bbox=t1_sc.bbox, padding=padding, on_crop=True))
            fig.show()
            while True:
                ans = input("1. over or 2. under 3. discard\n")
                try:
                    ans = int(ans)
                    break
                except Exception as e:
                    print("invalid input,", e)
                    fig.show()
                    continue

            print("selected: ", end="")
            if int(ans) == 1:
                print("<over>")
                over_maps.append(info)
            elif int(ans) == 2:
                print("<under>")
                under_maps.append(info)
            else:
                print("<discard>")
                discarded_maps.append(info)
            # plt.clf()
            # plt.cla()
            plt.close()
        return over_maps, under_maps, discarded_maps

    def save_map_data(mappings, path):
        """_summary_

        Parameters
        ----------
        mappings : _type_
            [t1, t2, t1_label, mapping]
        path : _type_
            _description_
        """
        json_data = []
        for smap in mappings:
            data = {
                "t1": smap[0],
                "t2": smap[1],
                "label": int(smap[2]),
                "mapping": {str(k): v for k, v in smap[3].items()},
            }
            json_data.append(data)

        with open(path, "w+") as f:
            json.dump(json_data, f)
        return json_data

    over_maps, under_maps, discarded_maps = human_loop_answer_over_under_seg(multiple_maps, time2id2sc)

    pos_data_dir = Path("./notebook_results/real_ebss_stav_data" / pos_path)
    os.makedirs(pos_data_dir, exist_ok=True)
    save_map_data(over_maps, pos_data_dir / ("overseg_maps_interval-%s.json" % sample_interval))
    save_map_data(under_maps, pos_data_dir / ("underseg_maps_interval-%s.json" % sample_interval))

    # %% [markdown]
    # COCO
    # ```
    # {
    #     "info": {...},
    #     "licenses": [...],
    #     "images": [...],
    #     "annotations": [...],
    #     "categories": [...], <-- Not in Captions annotations
    #     "segment_info": [...] <-- Only in Panoptic annotations
    # }
    #
    # "annotations": [
    #     {
    #         "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]],
    #         "area": 702.1057499999998,
    #         "iscrowd": 0,
    #         "image_id": 289343,
    #         "bbox": [473.07,395.93,38.65,28.67],
    #         "category_id": 18,
    #         "id": 1768
    #     },
    #     ...
    #     {
    #         "segmentation": {
    #             "counts": [179,27,392,41,…,55,20],
    #             "size": [426,640]
    #         },
    #         "area": 220834,
    #         "iscrowd": 1,
    #         "image_id": 250282,
    #         "bbox": [0,34,639,388],
    #         "category_id": 1,
    #         "id": 900100250282
    #     }
    # ]
    #
    # ```
    #

    # %%
    time_label2sc = {}
    for sc in single_cells:
        time_label2sc[(sc.timeframe, sc.id)] = sc

    # %%
    OVER_GT_CAT_ID = 0
    UNDER_GT_CAT_ID = 1
    OVER_CAT_ID = 2
    UNDER_CAT_ID = 3

    # %%
    # convert label maps to coco

    def ou_maps_to_coco(
        data,
        mask_dataset: LiveCellImageDataset,
        img_dataset: LiveCellImageDataset,
        mode,
        over_gt_cat_id=0,
        under_gt_cat_id=1,
        over_cat_id=2,
        under_cat_id=3,
    ):
        """save over/under segmentation maps to coco format.
        <associated_ann_id> key in annotation keys connects the over or under gt and wrong masks.
        1. over-segmentation cases, the over-segmentation (one mask) is the ground truth. All the wrong segmentations have the same <associated_ann_id> as the ground truth.
        2. under-segmentation cases, the under-segmentation (several masks) masks are the ground truth. All the CORRECT segmentations have the same <associated_ann_id> as the wrong masks.
        Four categories can be created:
        1. over-segmentation ground truth
        2. under-segmentation ground truth
        3. over-segmentation
        4. under-segmentation

        Parameters
        ----------
        data : _type_
            [(t1, t2, t1_label, mapping), ...)]
        mask_dataset : _type_
            _description_
        img_dataset : _type_
            _description_
        """

        def get_coco_contour_from_sc(sc):
            contour = np.copy(sc.contour)
            contour[:, 0], contour[:, 1] = sc.contour[:, 1], sc.contour[:, 0]
            contour = [list([list([int(coord) for coord in pos]) for pos in contour])]
            return contour

        res_coco = {
            "annotations": [],
            "images": [],
            "categories": [
                {"supercategory": "seg", "id": 0, "name": "gt"},
                {"supercategory": "seg", "id": 1, "name": "overseg"},
                {"supercategory": "seg", "id": 2, "name": "underseg"},
            ],
            "info": {
                "description": "over/under segmentation maps",
            },
        }

        times = list(img_dataset.time2url.keys())
        for time in times:
            img = img_dataset.get_img_by_time(time)
            res_coco["images"].append(
                {
                    "id": int(time),
                    "file_name": img_dataset.time2url[time],
                    "coco_url": img_dataset.time2url[time],
                    "height": int(img.shape[1]),
                    "width": int(img.shape[0]),
                }
            )

        if mode == "overseg":
            t1_cat_label = over_gt_cat_id
            t2_cat_label = over_cat_id
        elif mode == "underseg":
            t1_cat_label = under_cat_id
            t2_cat_label = under_gt_cat_id
        ann_id = 0
        for idx, (t1, t2, t1_label, mapping) in enumerate(data):
            sc = time_label2sc[(t1, t1_label)]
            img_url = img_dataset.time2url[sc.timeframe]
            img_id = int(sc.timeframe)
            res_coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": int(img_id),
                    "category_id": int(t1_cat_label),  # gt
                    "segmentation": get_coco_contour_from_sc(sc),
                    "label": int(t1_label),
                }
            )
            associated_ann_id = ann_id
            ann_id += 1
            for t2_label in mapping:
                tmp_sc = time_label2sc[(t2, t2_label)]
                res_coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": int(img_id),
                        "category_id": int(t2_cat_label),  # overseg
                        "segmentation": get_coco_contour_from_sc(tmp_sc),
                        "associated_ann_id": associated_ann_id,
                        "label": int(t2_label),
                    }
                )
                ann_id += 1
        return res_coco

    over_coco = ou_maps_to_coco(over_maps, label_mask_dataset, dic_dataset, "overseg")
    under_coco = ou_maps_to_coco(under_maps, label_mask_dataset, dic_dataset, "underseg")

    with open(pos_data_dir / ("coco_overseg_interval-%s.json" % sample_interval), "w+") as f:
        json.dump(over_coco, f)
    with open(pos_data_dir / ("coco_underseg_interval-%s.json" % sample_interval), "w+") as f:
        json.dump(under_coco, f)

    # %%
    len(over_maps), len(under_maps)

    # %%
    padding = 200
    sc.show_contour_mask(crop=True, padding=padding)

    contour_coords = sc.get_contour_coords_on_crop(padding=padding)
    plt.scatter(contour_coords[:, 1], contour_coords[:, 0], s=1, c="r")

    # %%

    # %%
    coco_data = COCO(pos_data_dir / ("coco_overseg_interval-%s.json" % sample_interval))
    overseg_scs = coco_to_sc(coco_data)

    # %%
    over_seg_gt_scs = []
    for sc in overseg_scs:
        if sc.meta["category_id"] == OVER_GT_CAT_ID:
            sc.uns["associated_scs"] = []
            for other_sc in overseg_scs:
                if "associated_ann_id" in other_sc.meta and other_sc.meta["associated_ann_id"] == sc.meta["id"]:
                    sc.uns["associated_scs"].append(other_sc)
                    assert other_sc.meta["category_id"] == OVER_CAT_ID  # must be over-segmentation cat id
            over_seg_gt_scs.append(sc)

    # %% [markdown]
    # check `over_seg_gt_scs`

    # %%

    # %%
    from livecell_tracker.segment.ou_utils import csn_augment_helper

    # %% [markdown]
    # Save overseg cases

    # %%
    subdir = Path("real_overseg_td1_" + str(pos_path))
    overseg_out_dir = out_dir / subdir
    raw_out_dir = overseg_out_dir / "raw"

    # seg_out_dir is the directory containing all raw segmentation masks for training
    # e.g. the eroded raw segmentation masks
    seg_out_dir = overseg_out_dir / "seg"

    # raw_seg_dir is the directory containing all raw segmentation masks for recording purposes
    raw_seg_dir = overseg_out_dir / "raw_seg_crop"
    gt_out_dir = overseg_out_dir / "gt"
    gt_label_out_dir = overseg_out_dir / "gt_label_mask"
    augmented_seg_dir = overseg_out_dir / "augmented_seg"
    raw_transformed_img_dir = overseg_out_dir / "raw_transformed_img"
    augmented_diff_seg_dir = overseg_out_dir / "augmented_diff_seg"
    meta_path = overseg_out_dir / "metadata.csv"
    overseg_df_save_path = overseg_out_dir / "data.csv"

    os.makedirs(raw_out_dir, exist_ok=True)
    os.makedirs(seg_out_dir, exist_ok=True)
    os.makedirs(raw_seg_dir, exist_ok=True)
    os.makedirs(gt_out_dir, exist_ok=True)
    os.makedirs(augmented_seg_dir, exist_ok=True)
    os.makedirs(gt_label_out_dir, exist_ok=True)
    os.makedirs(raw_transformed_img_dir, exist_ok=True)
    os.makedirs(augmented_diff_seg_dir, exist_ok=True)

    # %%
    # calculate label of scs
    # over_seg_gt_scs = parallelize(set_sc_label, [[sc] for sc in over_seg_gt_scs], cores=None)

    # %%
    import tqdm
    import pandas as pd

    overseg_erosion_scale_factors = np.linspace(-0.1, 0, 10)
    overseg_train_path_tuples = []
    augmented_overseg_data = []
    sample_id = 0

    all_df = None
    for sc in tqdm.tqdm(over_seg_gt_scs):
        img_crop = sc.get_img_crop()
        combined_gt_label_mask = sc.get_contour_mask()

        associated_scs = sc.uns["associated_scs"]
        seg_crop = np.zeros(combined_gt_label_mask.shape, dtype=np.uint8)
        assert len(associated_scs) > 0 and len(associated_scs) < 256, "number of associated scs must be in [1, 255]"
        for idx, other_sc in enumerate(associated_scs):
            other_seg_crop = other_sc.get_contour_mask(bbox=sc.bbox, crop=False)
            seg_crop[other_seg_crop > 0] = idx + 1

        raw_seg_crop = sc.get_img_crop()

        img_id = sc.meta["img_id"]
        filename_pattern = "img-%d_sample-%d.tif"
        raw_img_path = raw_out_dir / ("img-%d_sample-%d.tif" % (img_id, sample_id))
        seg_img_path = seg_out_dir / ("img-%d_sample-%d.tif" % (img_id, sample_id))
        gt_img_path = gt_out_dir / ("img-%d_sample-%d.tif" % (img_id, sample_id))
        gt_label_img_path = gt_label_out_dir / ("img-%d_sample-%d.tif" % (img_id, sample_id))
        raw_seg_img_path = raw_seg_dir / (filename_pattern % (img_id, sample_id))

        # fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        # axes[0].imshow(img_crop)
        # axes[0].set_title("img_crop")
        # axes[1].imshow(seg_crop)
        # axes[1].set_title("seg_crop")
        # axes[2].imshow(raw_seg_crop)
        # axes[2].set_title("raw_seg_crop")
        # axes[3].imshow(combined_gt_label_mask)
        # axes[3].set_title("combined_gt_label_mask")
        # plt.show()
        res_dict = csn_augment_helper(
            img_crop=img_crop,
            seg_label_crop=seg_crop,
            combined_gt_label_mask=combined_gt_label_mask,
            overseg_raw_seg_crop=raw_seg_crop,
            overseg_raw_seg_img_path=raw_seg_img_path,
            scale_factors=overseg_erosion_scale_factors,
            train_path_tuples=overseg_train_path_tuples,
            augmented_data=augmented_overseg_data,
            img_id=img_id,
            seg_label=sample_id,
            gt_label=None,  # t1 sc's label
            raw_img_path=raw_img_path,
            seg_img_path=seg_img_path,
            gt_img_path=gt_img_path,
            gt_label_img_path=gt_label_img_path,
            augmented_seg_dir=augmented_seg_dir,
            augmented_diff_seg_dir=augmented_diff_seg_dir,
            filename_pattern=filename_pattern,
            raw_transformed_img_dir=raw_transformed_img_dir,
            # df_save_path=overseg_df_save_path,
        )
        df = res_dict["df"]
        all_df = df  # because we pass train_path_tuples by reference, we don't need to concatenate df

    with open(overseg_df_save_path, "w+") as f:
        if all_df is not None:
            all_df.to_csv(f, index=False)

    # %% [markdown]
    # Save underseg cases

    # %%
    underseg_coco_data = COCO(pos_data_dir / ("coco_underseg_interval-%s.json" % sample_interval))
    underseg_scs = coco_to_sc(underseg_coco_data)

    # %%
    under_seg_gt_scs = []
    for sc in underseg_scs:
        if sc.meta["category_id"] == UNDER_GT_CAT_ID:
            sc.uns["associated_scs"] = []
            for other_sc in underseg_scs:
                if sc.meta["associated_ann_id"] == other_sc.meta["id"]:
                    sc.uns["associated_scs"].append(other_sc)
                    assert other_sc.meta["category_id"] == UNDER_CAT_ID  # must be over-segmentation cat id
            under_seg_gt_scs.append(sc)

    # %%
    underseg_sc2scs = {}
    for sc in under_seg_gt_scs:
        assert len(sc.uns["associated_scs"]) == 1, "under-segmentation gt sc must have exactly one associated sc"
        underseg_sc = sc.uns["associated_scs"][0]
        if underseg_sc in underseg_sc2scs:
            underseg_sc2scs[underseg_sc].append(sc)
        else:
            underseg_sc2scs[underseg_sc] = [sc]

    len(underseg_sc2scs)

    # %%
    from livecell_tracker.segment.ou_utils import csn_augment_helper, underseg_overlay_gt_masks, underseg_overlay_scs

    # %%
    from typing import Tuple

    # %%
    subdir = Path("real_underseg_td1_" + str(pos_path))
    underseg_out_dir = out_dir / subdir
    raw_out_dir = underseg_out_dir / "raw"

    # seg_out_dir is the directory containing all raw segmentation masks for training
    # e.g. the eroded raw segmentation masks
    seg_out_dir = underseg_out_dir / "seg"

    # raw_seg_dir is the directory containing all raw segmentation masks for recording purposes
    raw_seg_dir = underseg_out_dir / "raw_seg_crop"
    gt_out_dir = underseg_out_dir / "gt"
    gt_label_out_dir = underseg_out_dir / "gt_label_mask"
    augmented_seg_dir = underseg_out_dir / "augmented_seg"
    raw_transformed_img_dir = underseg_out_dir / "raw_transformed_img"
    augmented_diff_seg_dir = underseg_out_dir / "augmented_diff_seg"
    meta_path = underseg_out_dir / "metadata.csv"
    underseg_df_save_path = underseg_out_dir / "data.csv"

    os.makedirs(raw_out_dir, exist_ok=True)
    os.makedirs(seg_out_dir, exist_ok=True)
    os.makedirs(raw_seg_dir, exist_ok=True)
    os.makedirs(gt_out_dir, exist_ok=True)
    os.makedirs(augmented_seg_dir, exist_ok=True)
    os.makedirs(gt_label_out_dir, exist_ok=True)
    os.makedirs(raw_transformed_img_dir, exist_ok=True)
    os.makedirs(augmented_diff_seg_dir, exist_ok=True)

    scale_factors = np.linspace(0, 0.3, 10)
    train_path_tuples = []
    augmented_data = []
    all_df = None
    for sc in tqdm.tqdm(underseg_sc2scs):
        scs = underseg_sc2scs[sc]
        img_id, seg_label = int(sc.timeframe), int(sc.meta["label"])
        assert len(scs) > 0, "the list of single cells should not be empty"
        # sc.show_panel()
        (img_crop, seg_crop, combined_gt_label_mask) = underseg_overlay_scs(sc, scs, padding_scale=2)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_crop)
        axes[0].set_title("img_crop")
        axes[1].imshow(seg_crop)
        axes[1].set_title("seg_crop")
        axes[2].imshow(combined_gt_label_mask)
        axes[2].set_title("combined_gt_label_mask")

        raw_img_path = raw_out_dir / ("img-%d_seg-%d.tif" % (img_id, seg_label))
        seg_img_path = seg_out_dir / ("img-%d_seg-%d.tif" % (img_id, seg_label))
        gt_img_path = gt_out_dir / ("img-%d_seg-%d.tif" % (img_id, seg_label))
        gt_label_img_path = gt_label_out_dir / ("img-%d_seg-%d.tif" % (img_id, seg_label))

        # call csn augment helper
        res_dict = csn_augment_helper(
            img_crop=img_crop,
            seg_label_crop=seg_crop,
            combined_gt_label_mask=combined_gt_label_mask,
            scale_factors=scale_factors,
            train_path_tuples=train_path_tuples,
            augmented_data=augmented_data,
            img_id=img_id,
            seg_label=seg_label,
            gt_label=None,
            raw_img_path=raw_img_path,
            seg_img_path=seg_img_path,
            gt_img_path=gt_img_path,
            gt_label_img_path=gt_label_img_path,
            augmented_seg_dir=augmented_seg_dir,
            augmented_diff_seg_dir=augmented_diff_seg_dir,
            raw_transformed_img_dir=raw_transformed_img_dir,
        )
        all_df = res_dict["df"]

    if all_df is not None:
        all_df.to_csv(underseg_df_save_path)

    # %%
    from livecell_tracker.segment.ou_utils import csn_augment_helper

    # %%
    import pandas as pd

    dataframes = []
    for subdir in out_dir.iterdir():
        if subdir.is_dir():
            data_path = subdir / "data.csv"
            dataframe = pd.read_csv(data_path)
            dataframe["subdir"] = subdir.name
            dataframes.append(dataframe)
    combined_dataframe = pd.concat(dataframes)
    combined_dataframe.to_csv(out_dir / "train_data.csv", index=False)
