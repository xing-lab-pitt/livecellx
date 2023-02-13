import glob
import os
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm
from skimage import measure
from skimage.measure import regionprops
from livecell_tracker.core.datasets import LiveCellImageDataset, SingleImageDataset
from livecell_tracker.core.single_cell import SingleCellStatic


def get_contours_from_pred_masks(instance_pred_masks):
    # TODO add docs later
    contours = []
    for instance_mask in instance_pred_masks:
        tmp_contours = measure.find_contours(
            instance_mask, level=0.5, fully_connected="low", positive_orientation="low"
        )
        if len(tmp_contours) != 1:
            print("[WARN] more than 1 contour found in the instance mask")
        # convert to list for saving into json
        contours.extend([[list(coords) for coords in coord_arr] for coord_arr in tmp_contours])
    return contours


# TODO: docs
def match_mask_labels_by_iou(seg_label_mask, gt_label_mask, bg_label=0, return_all=False):
    """compute the similarity between ground truth mask and segmentation mask by intersection over union

    Parameters
    ----------
    seg_label_mask : _type_
        _description_
    gt_label_mask : _type_
        _description_
    bg_label : int, optional
        _description_, by default 0
    return_all : bool, optional
        _description_, by default False
    Returns
    -------
        A <gt2seg_map>, mapping ground truth keys to a dictionary of the best matching segmentation label and its iou
    """
    gt2seg_map = {}
    all_gt2seg_iou__map = {}
    # gets all the unique labels in the labeled_seg_mask and gtly_curated_mask
    seg_labels = np.unique(seg_label_mask)
    gt_labels = np.unique(gt_label_mask)

    temp_seg_mask = seg_label_mask.copy()
    temp_gt_mask = gt_label_mask.copy()

    for gt_label in gt_labels:
        if gt_label == bg_label:
            continue
        gt_label_key = gt_label
        all_gt2seg_iou__map[gt_label_key] = {}
        gt2seg_map[gt_label_key] = {}
        temp_gt_mask = gt_label_mask.copy()
        # isolates the current cell in the temp gtly_curated_mask and gets its pixels to 1
        temp_gt_mask[temp_gt_mask != gt_label] = 0
        temp_gt_mask[temp_gt_mask != 0] = 1

        best_iou = 0
        for seg_label in seg_labels:
            if seg_label == bg_label:
                continue
            temp_seg_mask = seg_label_mask.copy()

            # isolate the current cell in the temp_seg_mask and set its pixels to 1
            temp_seg_mask[temp_seg_mask != seg_label] = 0
            temp_seg_mask[temp_seg_mask != 0] = 1

            matching_rows, matching_columns = np.where(temp_seg_mask == 1)
            intersection_area = (temp_gt_mask[matching_rows, matching_columns] == 1).sum()
            union_area = temp_gt_mask.sum() + temp_seg_mask.sum() - intersection_area
            iou = intersection_area / union_area
            io_gt = intersection_area / temp_gt_mask.sum()
            io_seg = intersection_area / temp_seg_mask.sum()
            if gt_label_key not in all_gt2seg_iou__map:
                all_gt2seg_iou__map[gt_label_key] = []
            all_gt2seg_iou__map[gt_label_key].append(
                {
                    "seg_label": seg_label,
                    "iou": iou,
                    "io_gt": io_gt,
                    "io_seg": io_seg,
                }
            )

            if iou > best_iou:
                best_iou = iou
                gt2seg_map[gt_label_key]["best_iou"] = iou
                gt2seg_map[gt_label_key]["seg_label"] = seg_label
    if return_all:
        return gt2seg_map, all_gt2seg_iou__map
    else:
        return gt2seg_map


def mask_dataset_to_single_cells(mask_dataset: LiveCellImageDataset, img_dataset: LiveCellImageDataset):
    single_cells = []
    for time in mask_dataset.time2url:
        img = img_dataset.get_img_by_time(time)
        seg_mask = mask_dataset.get_img_by_time(time)
        props_list = regionprops(seg_mask)
        for prop in props_list:
            single_cells.append(
                SingleCellStatic(
                    timeframe=time,
                    img_dataset=img_dataset,
                    mask_dataset=mask_dataset,
                    bbox=prop.bbox,
                    contour=prop.coords,
                )
            )
    return single_cells
