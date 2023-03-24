from typing import List, Tuple
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


def create_ou_input_from_sc(sc: SingleCellStatic, padding_pixels: int = 0, dtype=float, remove_bg=True):
    if remove_bg:
        img_crop = sc.get_contour_img(padding=padding_pixels).astype(dtype)
    else:
        img_crop = sc.get_img_crop(padding=padding_pixels).astype(dtype)
    img_crop = normalize_img_to_uint8(img_crop).astype(dtype)
    img_crop[sc.get_contour_mask(padding=padding_pixels) == 0] *= -1
    return img_crop


def create_ou_input_from_imgs(img, mask, normalize=True, dtype=float):
    img_crop = img.astype(dtype)
    if normalize:
        img_crop = normalize_img_to_uint8(img_crop).astype(dtype)
    img_crop[mask == 0] *= -1
    return img_crop


# TODO: adapt to new sc API (check and fix the function below)
def underseg_overlay_gt_masks(
    seg_label: int, scs: List[SingleCellStatic], padding_scale=1.5, seg_mask=None
) -> Tuple[np.array, np.array, np.array]:
    """Overlay segmentation masks and ground truth masks for under-segmentation cases.
    Specifically, for a segmentation label, if there are multiple ground truth masks matched to it,
    then we overlay ground truths masks in the same mask

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
        seg_mask = scs[0].get_mask(dtype=int)

    seg_mask[seg_mask != int(seg_label)] = 0
    props_list = regionprops(seg_mask)

    if len(props_list) != 1:
        print(
            "[WARNING] skip: (%d, %d) due to more than one region found in seg mask or NO region found in seg mask"
            % (scs[0].timeframe, seg_label)
        )
        return None, None, None
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
    seg_crop = scs[0].get_mask_crop(bbox=seg_bbox, padding=padding_pixels, dtype=int)
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
        combined_gt_mask += (idx + 1) * sc.get_contour_mask(bbox=seg_bbox, padding=padding_pixels)
        img_crop = sc.get_img_crop(padding=padding_pixels) if img_crop is None else img_crop  # set img_crop once
        sc.bbox = tmp
    return (img_crop, seg_crop, combined_gt_mask)


def underseg_overlay_scs(
    underseg_sc: int, scs: SingleCellStatic, padding_scale=1.5, seg_mask=None
) -> Tuple[np.array, np.array, np.array]:
    """Overlay segmentation masks and ground truth masks for under-segmentation cases.
    Specifically, for a segmentation label, if there are multiple ground truth masks matched to it,
    then we overlay ground truths masks in the same mask

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

    # TODO: we may remove the check part below
    seg_mask = underseg_sc.get_contour_mask().astype(int)
    props_list = regionprops(seg_mask)
    if len(props_list) != 1:
        print(
            "[WARNING] skip: (time:%) due to more than one region found in seg mask or NO region found in seg mask. #props: %d"
            % (scs[0].timeframe, len(props_list))
        )
        print(np.unique(scs[0].get_mask(dtype=int)))
        return
    # #obtain segmentation bbox from segmentation mask
    # seg_props = props_list[0]
    # seg_bbox = seg_props.bbox

    seg_bbox = underseg_sc.bbox
    xmin, ymin, xmax, ymax = seg_bbox

    # compute padding based on scale
    padding_pixels = np.array((padding_scale - 1) * max(xmax - xmin, ymax - ymin))
    padding_pixels = padding_pixels.astype(int)

    # get seg mask's crop with single cell's get_mask_crop implementation for consistency
    seg_crop = underseg_sc.get_contour_mask(bbox=seg_bbox, padding=padding_pixels).astype(int)

    combined_gt_mask = np.zeros(seg_crop.shape)
    img_crop = None
    for idx, sc in enumerate(scs):
        combined_gt_mask += (idx + 1) * sc.get_contour_mask(bbox=seg_bbox, padding=padding_pixels)
        img_crop = (
            sc.get_img_crop(padding=padding_pixels, bbox=seg_bbox) if img_crop is None else img_crop
        )  # set img_crop once
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


def dilate_or_erode_label_mask(label_mask: np.array, scale_factor, bg_val=0):
    import scipy

    label_mask = label_mask.astype(np.uint8)
    labels = set(np.unique(label_mask))
    labels.remove(bg_val)
    res_mask = np.zeros_like(label_mask)
    for label in labels:
        tmp_bin_mask = (label_mask == label).astype(np.uint8)
        tmp_scaled_mask = dilate_or_erode_mask(tmp_bin_mask, scale_factor)
        res_mask = np.maximum(res_mask, tmp_scaled_mask)
    return res_mask


def csn_augment_helper(
    img_crop,
    seg_label_crop,
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
    """_summary_

    Parameters
    ----------
    img_crop : _type_
        _description_
    seg_label_crop : _type_
        In overseg case, this should be a label mask and each label region will be dilated/eroded correspondingly
        In underseg case, this can be a binary mask and should be dilated
    combined_gt_label_mask : _type_
        _description_
    scale_factors : list
        _description_
    train_path_tuples : list
        _description_
    augmented_data : list
        _description_
    img_id : _type_
        _description_
    seg_label : _type_
        _description_
    gt_label : _type_
        _description_
    raw_img_path : _type_
        _description_
    seg_img_path : _type_
        _description_
    gt_img_path : _type_
        _description_
    gt_label_img_path : _type_
        _description_
    augmented_seg_dir : _type_
        _description_
    augmented_diff_seg_dir : _type_
        _description_
    filename_pattern : str, optional
        _description_, by default "img-%d_seg-%d.tif"
    overseg_raw_seg_crop : _type_, optional
        _description_, by default None
    overseg_raw_seg_img_path : _type_, optional
        _description_, by default None
    raw_transformed_img_dir : _type_, optional
        _description_, by default None
    df_save_path : _type_, optional
        _description_, by default None
    normalize_img_uint8 : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    if train_path_tuples is None:
        train_path_tuples = []
    if normalize_img_uint8:
        img_crop = normalize_img_to_uint8(img_crop)
    combined_gt_binary_mask = combined_gt_label_mask > 0
    combined_gt_binary_mask = combined_gt_binary_mask.astype(np.uint8)

    save_tiff(img_crop, raw_img_path, mode="I")  # save to 32-bit depth signed integer
    save_tiff(seg_label_crop, seg_img_path)
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

        if np.unique(seg_label_crop).shape[0] > 256:
            print("[WARNING] skip: (%d, %d) due to more than 256 unique seg labels" % (img_id, seg_label))
            continue
        seg_label_crop = seg_label_crop.astype(np.uint8)

        # seg_crop should only contains one label
        # TODO: the condition commented above should be a postcondition of underseg_overlay_gt_masks
        # seg_label_crop[seg_label_crop > 0] = 1
        # aug_seg_crop = dilate_or_erode_mask(seg_label_crop, scale_factor=scale)
        aug_seg_crop = dilate_or_erode_label_mask(seg_label_crop, scale_factor=scale)
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
                "seg_crop": seg_label_crop,
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


def collect_and_combine_data(out_dir: Path):
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    dataframes = []
    for subdir in out_dir.iterdir():
        if subdir.is_dir():
            data_path = subdir / "data.csv"
            dataframe = pd.read_csv(data_path)
            dataframe["subdir"] = subdir.name
            dataframes.append(dataframe)
    combined_dataframe = pd.concat(dataframes)
    combined_dataframe.to_csv(out_dir / "train_data.csv", index=False)
