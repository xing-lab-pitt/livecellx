import glob
import os
import os.path
from pathlib import Path
from typing import Tuple
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm
from skimage import measure
from skimage.measure import regionprops
from multiprocessing import Pool
from skimage.measure import regionprops, find_contours

from livecellx.segment.ou_simulator import find_contours_opencv

from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset
from livecellx.core.single_cell import SingleCellStatic


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
        all_gt2seg_iou__map[gt_label_key] = []
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


def filter_labels_match_map(gt2seg_iou__map, iou_threshold):
    label_map = {}
    for label_1 in gt2seg_iou__map:
        label_map[label_1] = {}
        for score_info in gt2seg_iou__map[label_1]:
            if score_info["iou"] > iou_threshold:
                label_map[label_1][score_info["seg_label"]] = {"iou": score_info["iou"]}
    return label_map


def compute_match_label_map(t1, t2, mask_dataset, iou_threshold=0.2) -> tuple:
    """
    Compute the label map (mapping between objects) between two time points

    Parameters
    ----------
    t1 : _type_
        _description_
    t2 : _type_
        _description_
    mask_dataset : _type_
        _description_
    iou_threshold : float, optional
        _description_, by default 0.2

    Returns
    -------
    tuple
        A tuple consisting of 3 elements:
        - t1
        - t2
        - a dictionary of the form::

            {
                t1_label_1: {
                    t2_label_1: {
                        "iou": iou_score
                    },
                    t2_label_2: {
                        "iou": iou_score
                    },
                    ...
                },
                t1_label_2: {
                    t2_label_1: {
                        "iou": iou_score
                    },
                },
                ...
            }

    """
    label_mask1 = mask_dataset.get_img_by_time(t1)
    label_mask2 = mask_dataset.get_img_by_time(t2)

    # Note: first arg is mask2 and second arg is mask1 to create a label map from mask1 label to mask2
    # read match_mask_labels_by_iou docstring for more info
    _, score_dict = match_mask_labels_by_iou(label_mask2, label_mask1, return_all=True)
    label_map = {}
    for label_1 in score_dict:
        label_map[label_1] = {}
        for score_info in score_dict[label_1]:
            if score_info["iou"] > iou_threshold:
                label_map[label_1][score_info["seg_label"]] = {"iou": score_info["iou"]}
    return t1, t2, label_map


def process_scs_from_one_label_mask(label_mask_dataset, img_dataset, time, bg_val=0):
    label_mask = label_mask_dataset.get_img_by_time(time)
    labels = set(np.unique(label_mask))
    if bg_val in labels:
        labels.remove(bg_val)
    contours = []
    for label in labels:
        bin_mask = (label_mask == label).astype(np.uint8)
        label_contours = find_contours_opencv(bin_mask)
        assert len(label_contours) == 1
        contours.append(label_contours[0])

    # contours = find_contours(seg_mask) # skimage: find_contours
    _scs = []
    for contour in contours:
        _scs.append(
            SingleCellStatic(
                timeframe=time,
                img_dataset=img_dataset,
                mask_dataset=label_mask_dataset,
                contour=contour,
            )
        )
    return _scs


def process_mask_wrapper(args):
    return process_scs_from_one_label_mask(*args)


def prep_scs_from_mask_dataset(mask_dataset, dic_dataset, cores=None):
    scs = []
    inputs = [(mask_dataset, dic_dataset, time) for time in mask_dataset.time2url.keys()]
    pool = Pool(processes=cores)
    for _scs in tqdm(pool.imap_unordered(process_mask_wrapper, inputs), total=len(inputs)):
        scs.extend(_scs)
    pool.close()
    pool.join()
    return scs


def judge_connected_bfs(mask: np.ndarray, label1: int, label2: int) -> Tuple[bool, int]:
    def _is_valid(x: int, y: int, rows: int, cols: int) -> bool:
        return 0 <= x < rows and 0 <= y < cols

    rows, cols = mask.shape
    start = None
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] == label1:
                start = (i, j)
                break
        if start is not None:
            break

    if start is None:
        return False, 0

    visited = np.zeros_like(mask, dtype=bool)
    queue = deque([(start[0], start[1], 0)])
    visited[start] = True
    connected_pixels = 0

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        x, y, d = queue.popleft()

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if _is_valid(nx, ny, rows, cols) and not visited[nx, ny]:
                if mask[nx, ny] == label2:
                    connected_pixels += 1
                elif mask[nx, ny] == label1:
                    queue.append((nx, ny, d + 1))

                visited[nx, ny] = True

    return connected_pixels > 0, connected_pixels
