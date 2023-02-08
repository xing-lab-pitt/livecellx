from typing import Tuple
from skimage.measure import regionprops
import pandas as pd
from livecell_tracker.core.io_utils import save_tiff
import numpy as np
import json
from livecell_tracker.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from livecell_tracker.segment.detectron_utils import (
    convert_detectron_instance_pred_masks_to_binary_masks,
    convert_detectron_instances_to_label_masks,
    segment_images_by_detectron,
    segment_single_img_by_detectron_wrapper,
)
from livecell_tracker.core.datasets import LiveCellImageDataset
from livecell_tracker.preprocess.utils import (
    overlay,
    enhance_contrast,
    normalize_img_to_uint8,
)
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from livecell_tracker.preprocess.utils import dilate_or_erode_mask


def underseg_overlay_gt_masks(
    seg_label: int, scs: SingleCellStatic, padding_scale=1.5, seg_mask=None
) -> Tuple[np.array, np.array, np.array]:
    """Overlay segmentation masks and ground truth masks for under-segmentation cases.
    Specifically, for a segmentation label, if there are multiple ground truth masks matched to it,
    then we overlay ground truths masks in the same kmask

    Parameters
    ----------
    seg_label : int
        _description_
    scs : SingleCellStatic
        _description_
    padding_scale : float, optional
        _description_, by default 1.5
    mask :
        if not None, use the mask, otherwise inferred from other args, by default None

    Returns
    -------
    Tuple[np.array, np.array, np.array]
        (img_crop, seg_crop, combined ground-truth mask)
    """
    if len(scs) == 0:
        print("no scs for this seg_label")
        return None, None, None

    if seg_mask is None:
        seg_mask = scs[0].get_mask()

    seg_mask[seg_mask != seg_label] = 0
    props_list = regionprops(seg_mask)

    if len(props_list) != 1:
        print(
            "[WARNING] skip: (%d, %d) due to more than one region found in seg mask or NO region found in seg mask"
            % (scs[0].timeframe, seg_label)
        )
        return
    # obtain segmentation bbox from segmentation mask
    seg_props = props_list[0]
    seg_bbox = seg_props.bbox
    xmin, ymin, xmax, ymax = seg_bbox

    # compute padding based on scale
    padding_pixels = np.array((padding_scale - 1) * max(xmax - xmin, ymax - ymin))
    padding_pixels = padding_pixels.astype(int)

    # get seg mask's crop with single cell's get_mask_crop implementation for consistency
    tmp = np.array(scs[0].bbox)
    scs[0].bbox = seg_bbox
    seg_crop = scs[0].get_mask_crop(padding=padding_pixels)
    scs[0].bbox = np.array(tmp)

    # clear other seg labels
    seg_crop[seg_crop != seg_label] = 0
    seg_crop[seg_crop > 0] = 1

    combined_gt_mask = np.zeros(seg_crop.shape)
    img_crop = None
    for idx, sc in enumerate(scs):
        sc.meta["seg_label"] = None
        tmp = np.array(sc.bbox)
        sc.bbox = seg_bbox
        combined_gt_mask += (idx + 1) * sc.get_contour_mask(padding=padding_pixels)
        img_crop = sc.get_img_crop(padding=padding_pixels) if img_crop is None else img_crop  # set img_crop once
        sc.bbox = tmp
    return (img_crop, seg_crop, combined_gt_mask)


def gen_aug_diff_mask(aug_mask: np.array, combined_gt_mask: np.array) -> np.array:
    """generate a mask based on the difference between the augmented mask and the combined gt mask
    0: no difference
    -1: augmented mask is 0, combined gt mask is 1 -> over-segmentation
    1: augmented mask is 1, combined gt mask is 0 -> under-segmentation
    Note: special care for uint8 case when calculating difference mask if we use cv2 related functions

    Parameters
    ----------
    aug_mask : np.array
        _description_
    combined_gt_mask : np.array
        _description_

    Returns
    -------
    np.array
        _description_
    """
    aug_mask = aug_mask.astype(int)  # prevent uint8 overflow (-1 in diff case below)
    underseg_mask = np.zeros(aug_mask.shape)
    underseg_mask[combined_gt_mask > 0] = 1
    combined_gt_mask = combined_gt_mask.astype(int)
    diff_mask = aug_mask - combined_gt_mask  # should only contain 0 and 1
    assert len(np.unique(diff_mask)) <= 3

    return diff_mask


def csn_augment_helper(
    img_crop,
    seg_crop,
    combined_gt_label_mask,
    scale_factors: list,
    train_path_tuples: list,
    augmented_data: list,
    img_id,
    seg_label,
    gt_label,
    raw_img_path,
    seg_img_path,
    gt_img_path,
    gt_label_img_path,
    augmented_seg_dir,
    augmented_diff_seg_dir,
    filename_pattern="img-%d_seg-%d.tif",
    overseg_raw_seg_crop=None,
    overseg_raw_seg_img_path=None,
    raw_transformed_img_dir=None,
    df_save_path=None,
    normalize_img_uint8=True,
):

    if normalize_img_uint8:
        img_crop = normalize_img_to_uint8(img_crop)
    combined_gt_binary_mask = combined_gt_label_mask > 0
    combined_gt_binary_mask = combined_gt_binary_mask.astype(np.uint8)

    save_tiff(img_crop, raw_img_path, mode="I")  # save to 32-bit depth signed integer
    save_tiff(seg_crop, seg_img_path)
    save_tiff(combined_gt_binary_mask, gt_img_path)
    save_tiff(combined_gt_label_mask, gt_label_img_path)

    if overseg_raw_seg_img_path is not None:
        save_tiff(overseg_raw_seg_crop, overseg_raw_seg_img_path)

    # append aug-%d to filename pattern
    filename_root, ext = os.path.splitext(filename_pattern)
    aug_filename_pattern = filename_root + "_aug-%d" + ext
    # dilate or erode segmentation mask
    for idx, scale in enumerate(scale_factors):
        augmented_seg_path = augmented_seg_dir / (aug_filename_pattern % (img_id, seg_label, idx))
        augmented_diff_seg_path = augmented_diff_seg_dir / (aug_filename_pattern % (img_id, seg_label, idx))

        if np.unique(seg_crop).shape[0] > 256:
            print("[WARNING] skip: (%d, %d) due to more than 256 unique seg labels" % (img_id, seg_label))
            continue
        seg_crop = seg_crop.astype(np.uint8)

        # seg_crop should only contains one label
        # TODO: the condition commented above should be a postcondition of underseg_overlay_gt_masks
        seg_crop[seg_crop > 0] = 1
        aug_seg_crop = dilate_or_erode_mask(seg_crop, scale_factor=scale)
        aug_values = np.unique(aug_seg_crop)
        assert len(aug_values) <= 2, "only two values should be present in aug masks"
        aug_seg_crop[aug_seg_crop > 0] = 1
        aug_seg_crop[aug_seg_crop < 0] = 0  # not necessary, check math
        save_tiff(aug_seg_crop, augmented_seg_path)

        aug_diff_mask = gen_aug_diff_mask(aug_seg_crop, combined_gt_binary_mask)
        save_tiff(aug_diff_mask, augmented_diff_seg_path, mode="I")

        raw_transformed_img_path = raw_transformed_img_dir / ("img-%d_seg-%d_aug-%d.tif" % (img_id, seg_label, idx))
        raw_transformed_img_crop = img_crop.copy().astype(int)
        raw_transformed_img_crop[aug_seg_crop == 0] *= -1
        save_tiff(raw_transformed_img_crop, raw_transformed_img_path, mode="I")

        train_path_tuples.append(
            (
                raw_img_path.as_posix(),
                augmented_seg_path.as_posix(),
                gt_img_path.as_posix(),
                seg_img_path.as_posix(),
                scale,
                augmented_diff_seg_path.as_posix(),
                gt_label_img_path.as_posix(),
                raw_transformed_img_path.as_posix(),
            )
        )

        augmented_data.append(
            {
                "img_id": img_id,
                "img_crop": img_crop,
                "seg_crop": seg_crop,
                "seg_label": seg_label,
                "gt_label": gt_label,
                "combined_gt_mask": combined_gt_binary_mask,
                "aug_seg_crop": aug_seg_crop,
                "aug_diff_mask": aug_diff_mask,
                "combined_gt_label_mask": combined_gt_label_mask,
                "raw_transformed_img_crop": raw_transformed_img_crop,
            }
        )
        # augmented_data[(img_id, seg_label)].append(
        #     (
        #         img_crop,
        #         seg_crop,
        #         combined_gt_mask,
        #         aug_seg_crop,
        #         aug_diff_mask,
        #         combined_gt_label_mask,
        #         raw_transformed_img_crop,
        #     )
        # )
    cols = ["raw", "seg", "gt", "raw_seg", "scale", "aug_diff_mask", "gt_label_mask", "raw_transformed_img"]
    df = pd.DataFrame(train_path_tuples, columns=cols)

    # when generate samples in parallel mode, we need to double check race conditions
    # currently disable saving when generating samples in parallel mode
    if df_save_path:
        if os.path.exists(df_save_path):
            df.to_csv(
                df_save_path,
                index=False,
                mode="a",
                header=False,
            )
        else:
            df.to_csv(df_save_path, index=False)

    return {
        "train_path_tuples": train_path_tuples,
        "augmented_data": augmented_data,
        "df": df,
    }
