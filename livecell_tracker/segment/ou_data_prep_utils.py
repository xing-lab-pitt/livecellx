import pandas as pd
import os
from pathlib import Path
from livecell_tracker.core.datasets import read_img_default
from livecell_tracker.segment.ou_utils import csn_augment_helper
import numpy as np
import skimage.measure
import shutil


def extend_overseg_subdir(df_path, overseg_out_dir: Path):
    RAW_COL = "raw"
    SEG_COL = "seg"
    RAW_SEG_COL = "raw_seg"
    SCALE_COL = "scale"
    GT_COL = "gt"
    GT_LABEL_COL = "gt_label_mask"
    if overseg_out_dir.exists():
        print("Deleting existing directory: ", overseg_out_dir)
        shutil.rmtree(overseg_out_dir, ignore_errors=True)
    df = pd.read_csv(df_path)
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
    syn_overseg_df_save_path = overseg_out_dir / "data.csv"

    os.makedirs(raw_out_dir, exist_ok=True)
    os.makedirs(seg_out_dir, exist_ok=True)
    os.makedirs(raw_seg_dir, exist_ok=True)
    os.makedirs(gt_out_dir, exist_ok=True)
    os.makedirs(augmented_seg_dir, exist_ok=True)
    os.makedirs(gt_label_out_dir, exist_ok=True)
    os.makedirs(raw_transformed_img_dir, exist_ok=True)
    os.makedirs(augmented_diff_seg_dir, exist_ok=True)

    overseg_train_path_tuples = []
    augmented_overseg_data = []
    filename_pattern = "img-%d-seg-%d.png"
    overseg_metadata = []
    overseg_erosion_scale_factors = np.linspace(-0.1, 0, 10)
    all_df = None

    sample_id = 0
    for row_idx in range(len(df)):
        orig_raw_path = df[RAW_COL][row_idx]
        orig_seg_path = df[SEG_COL][row_idx]
        orig_raw_seg_path = df[RAW_SEG_COL][row_idx]
        orig_gt_label_path = df[GT_LABEL_COL][row_idx]

        # load raw image
        img_crop = read_img_default(orig_raw_path)
        # load segmentation mask
        seg_crop = read_img_default(orig_seg_path)
        # load raw segmentation mask
        raw_seg_img = read_img_default(orig_raw_seg_path)
        # load ground truth label mask
        combined_gt_label_mask = read_img_default(orig_gt_label_path)

        assert img_crop.shape == seg_crop.shape == combined_gt_label_mask.shape
        filename = filename_pattern % (sample_id, sample_id)
        raw_img_path = raw_out_dir / filename
        seg_img_path = seg_out_dir / filename
        raw_seg_img_path = raw_seg_dir / filename
        gt_img_path = gt_out_dir / filename
        gt_label_img_path = gt_label_out_dir / filename

        # # metadata is a dict, containing params used to genereate our synthetic overseg data
        # meta_info["raw_img_path"] = raw_img_path
        # meta_info["seg_img_path"] = seg_img_path
        # meta_info["gt_img_path"] = gt_img_path

        # overseg_metadata.append(meta_info)

        # call csn augment helper
        label_seg_crop = skimage.measure.label(seg_crop)
        labels = np.unique(label_seg_crop)
        # remove bg label
        labels = labels[labels != 0]

        # for each label, generate data based on label only pixels
        for label in labels:
            # get label mask
            specific_seg_label_mask = (label_seg_crop == label).astype(np.uint8)
            sample_id += 1
            res_dict = csn_augment_helper(
                img_crop=img_crop,
                seg_label_crop=specific_seg_label_mask,
                combined_gt_label_mask=combined_gt_label_mask,
                overseg_raw_seg_crop=specific_seg_label_mask,
                overseg_raw_seg_img_path=raw_seg_img_path,
                scale_factors=overseg_erosion_scale_factors,
                train_path_tuples=overseg_train_path_tuples,
                augmented_data=augmented_overseg_data,
                img_id=sample_id,
                seg_label=sample_id,
                gt_label=sample_id,
                raw_img_path=raw_img_path,
                seg_img_path=seg_img_path,
                gt_img_path=gt_img_path,
                gt_label_img_path=gt_label_img_path,
                augmented_seg_dir=augmented_seg_dir,
                augmented_diff_seg_dir=augmented_diff_seg_dir,
                filename_pattern=filename_pattern,
                raw_transformed_img_dir=raw_transformed_img_dir,
                # df_save_path=syn_overseg_df_save_path,
            )
            all_df = res_dict["df"]

    all_df.to_csv(syn_overseg_df_save_path, index=False)
