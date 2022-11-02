import glob
import os
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm
from skimage import measure


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
def match_mask_labels_by_iou(seg_mask, gt_mask, bg_label=0):
    """compute the similarity between segmentation mask and ground truth mask by intersection over union

    Parameters
    ----------
    seg_mask : _type_
        _description_
    gt_mask : _type_
        _description_
    bg_label : int, optional
        _description_, by default 0

    Returns
    -------
    _type_
        _description_
    """
    manual2seg_map = {}
    # gets all the unique labels in the labeled_seg_mask and manually_curated_mask
    seg_labels = np.unique(seg_mask)
    gt_labels = np.unique(gt_mask)

    temp_seg_mask = seg_mask.copy()
    temp_gt_mask = gt_mask.copy()

    # loops the cells in the manually_curated_mask
    for gt_label in gt_labels:
        if gt_label == bg_label:
            continue
        temp_gt_mask = gt_mask.copy()
        # isolates the current cell in the temp manually_curated_mask and gets its pixels to 1
        temp_gt_mask[temp_gt_mask != gt_label] = 0
        temp_gt_mask[temp_gt_mask != 0] = 1

        best_iou = 0
        manual_label_key = gt_label
        manual2seg_map[manual_label_key] = {}
        for seg_label in seg_labels:
            if seg_label == bg_label:
                continue
            temp_seg_mask = seg_mask.copy()
            # isolates the current cell in the temp_seg_mask and gets its pixels to 1
            temp_seg_mask[temp_seg_mask != seg_label] = 0
            temp_seg_mask[temp_seg_mask != 0] = 1

            matching_rows, matching_columns = np.where(temp_seg_mask == 1)
            intersection_area = (temp_gt_mask[matching_rows, matching_columns] == 1).sum()
            union_area = temp_gt_mask.sum() + temp_seg_mask.sum() - intersection_area
            iou = intersection_area / union_area
            if iou > best_iou:
                best_iou = iou
                manual2seg_map[manual_label_key]["best_iou"] = iou
                manual2seg_map[manual_label_key]["seg_label"] = seg_label
    return manual2seg_map
