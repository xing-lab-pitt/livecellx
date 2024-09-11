import argparse
import glob
import gzip
import json
import os.path
import sys
import time
from collections import deque
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch import Tensor
from torch.nn import init
from torch.utils.data import DataLoader, random_split
import scipy.ndimage
import skimage.measure
from livecellx.core.utils import label_mask_to_edt_mask
from livecellx.preprocess.utils import normalize_img_to_uint8, normalize_edt

# class CorrectSegNetData(data.Dataset):
#     def __init__(self, livecell_dataset: LiveCellImageDataset, segnet_dataset: LiveCellImageDataset):
#         self.livecell_dataset = livecell_dataset
#         self.segnet_dataset = segnet_dataset


class CorrectSegNetDataset(torch.utils.data.Dataset):
    """Dataset for training CorrectSegNetDatasert"""

    OVERSEG_ONEHOT = [1, 0, 0, 0]
    UNDERSEG_ONEHOT = [0, 1, 0, 0]
    DROPOUT_ONEHOT = [0, 0, 1, 0]
    CORRECT_ONEHOT = [0, 0, 0, 1]

    def __init__(
        self,
        raw_img_paths: List[str],
        seg_mask_paths: List[str],
        gt_mask_paths: List[str],
        gt_label_mask_paths: List[str],
        raw_seg_paths: List[str],
        scales: List[float],
        transform=None,
        raw_transformed_img_paths: List[str] = None,
        aug_diff_img_paths: List[str] = None,
        input_type="raw_aug_seg",
        apply_gt_seg_edt=False,
        exclude_raw_input_bg=False,
        subdirs=None,
        raw_df=None,
        normalize_uint8=False,
        bg_val=0,
        use_gt_pixel_weight=False,
        force_no_edt_aug=False,
    ):
        """_summary_

        Parameters
        ----------
        raw_img_paths : List[str]
            _description_
        seg_mask_paths : List[str]
            _description_
        gt_mask_paths : List[str]
            _description_
        raw_seg_paths : List[str]
            _description_
        scales : List[float]
            _description_
        transform : _type_, optional
            _description_, by default None
        raw_transformed_img_paths : List[str], optional
            _description_, by default None
        aug_diff_img_paths : List[str], optional
            _description_, by default None
        input_type : str, optional
            _description_, by default "raw_aug_seg"
        apply_gt_seg_edt : bool, optional
            _description_, by default False
        exclude_raw_bg : bool, optional
            if True, exclude all background pixels (including cells in bg) in input, by default False
        """
        self.raw_img_paths = raw_img_paths
        self.scaled_seg_mask_paths = seg_mask_paths
        self.gt_mask_paths = gt_mask_paths
        self.gt_label_mask_paths = gt_label_mask_paths
        self.transform = transform
        self.raw_seg_paths = raw_seg_paths
        self.raw_transformed_img_paths = raw_transformed_img_paths
        self.aug_diff_img_paths = aug_diff_img_paths

        self.scales = scales
        assert (
            len(self.raw_img_paths) == len(self.scaled_seg_mask_paths) == len(self.gt_mask_paths)
        ), "The number of images, segmentation masks and ground truth masks must be the same."
        self.input_type = input_type
        self.apply_gt_seg_edt = apply_gt_seg_edt
        self.exclude_raw_input_bg = exclude_raw_input_bg
        if subdirs is None and raw_df is not None:
            self.subdirs = raw_df["subdir"].values
        else:
            self.subdirs = subdirs
        self.subdir_set = set(self.subdirs)
        self.raw_df = raw_df
        print("input type:", self.input_type)
        print("if apply_gt_seg_edt:", self.apply_gt_seg_edt)

        self.normalize_uint8 = normalize_uint8
        print("whether to normalize_uint8:", self.normalize_uint8)
        self.bg_val = bg_val

        self.use_gt_pixel_weight = use_gt_pixel_weight
        print("whether to use_gt_pixel_weight:", self.use_gt_pixel_weight)
        if self.use_gt_pixel_weight:
            self.gt_pixel_weight_paths = [
                str(Path(path).parent.parent / "gt_pixel_weight" / (str(Path(path).stem) + "_weight.npy"))
                for path in self.gt_mask_paths
            ]

        self.force_no_edt_aug = force_no_edt_aug

    def get_raw_seg(self, idx) -> np.array:
        return np.array(Image.open(self.raw_seg_paths[idx]))

    def get_scale(self, idx):
        return self.scales[idx]

    def get_subdir(self, idx):
        return self.subdirs.iloc[idx]

    def label_mask_to_edt(label_mask: np.array, bg_val=0):
        label_mask = label_mask.astype(np.uint8)
        labels = set(np.unique(label_mask))
        labels.remove(bg_val)
        res_edt = np.zeros_like(label_mask)
        for label in labels:
            tmp_bin_mask = label_mask == label
            tmp_edt = scipy.ndimage.morphology.distance_transform_edt(tmp_bin_mask)
            res_edt = np.maximum(res_edt, tmp_edt)
        return res_edt

    def __getitem__(self, idx):
        """
        Get the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the following keys:
                - "input": The input image tensor.
                - "seg_mask": The segmented mask tensor. If input is edt_v0, this is the edt version.
                - "gt_mask_binary": The binary ground truth mask tensor.
                - "gt_mask": The combined ground truth tensor.
                - "idx": The index of the item.
                - "gt_label_mask": The ground truth label mask tensor.
                - "ou_aux": The auxiliary output tensor.
                - "gt_pixel_weight": The ground truth pixel weight tensor.
                - "gt_mask_edt" (optional): The ground truth mask tensor with Euclidean Distance Transform applied.
        """
        augmented_raw_img = Image.open(self.raw_img_paths[idx])
        scaled_seg_mask = Image.open(self.scaled_seg_mask_paths[idx])
        gt_mask = Image.open(self.gt_mask_paths[idx])
        augmented_raw_transformed_img = Image.open(self.raw_transformed_img_paths[idx])
        aug_diff_img = Image.open(self.aug_diff_img_paths[idx])
        gt_label_mask__np = np.array(Image.open(self.gt_label_mask_paths[idx]))
        if "ou_aux" in self.raw_df.columns:
            ou_aux_label = self.raw_df["ou_aux"].iloc[idx]
        else:
            ou_aux_label = None
        if self.normalize_uint8:
            augmented_raw_img = normalize_img_to_uint8(np.array(augmented_raw_img))
            augmented_raw_transformed_img = normalize_img_to_uint8(np.array(augmented_raw_transformed_img))

        augmented_raw_img = torch.tensor(np.array(augmented_raw_img)).float()
        scaled_seg_mask = torch.tensor(np.array(scaled_seg_mask)).float()
        gt_mask = torch.tensor(np.array(gt_mask)).long()
        augmented_raw_transformed_img = torch.tensor(np.array(augmented_raw_transformed_img)).float()
        aug_diff_img = torch.tensor(np.array(aug_diff_img)).float()
        gt_label_mask = torch.tensor(gt_label_mask__np.copy()).long()

        if self.use_gt_pixel_weight:
            # Read the pixel weight map from the <gt_pixel_weight> subfolder. weights are in npy format
            gt_pixel_weight = np.load(self.gt_pixel_weight_paths[idx])
        else:
            # Ones for all pixels
            gt_pixel_weight = np.ones_like(gt_label_mask__np)
        gt_pixel_weight = torch.tensor(gt_pixel_weight).float()

        # Transform to edt for inputs before augmentation
        if self.input_type == "edt_v0":
            scaled_seg_mask = label_mask_to_edt_mask(scaled_seg_mask, bg_val=self.bg_val)
            scaled_seg_mask = torch.tensor(scaled_seg_mask).float()

        gt_label_edt = torch.tensor(label_mask_to_edt_mask(gt_label_mask, bg_val=self.bg_val)).float()

        # Prepare for augmentation
        concat_img = torch.stack(
            [
                augmented_raw_img,
                augmented_raw_transformed_img,
                scaled_seg_mask,
                gt_mask.float(),
                aug_diff_img,
                gt_label_mask,
                gt_pixel_weight,
                gt_label_edt,
            ],
            dim=0,
        )

        if self.transform:
            concat_img = self.transform(concat_img)

        augmented_raw_img = concat_img[0]
        augmented_raw_transformed_img = concat_img[1]
        augmented_scaled_seg_mask = concat_img[2]
        augmented_gt_label_mask = concat_img[5].long()
        augmented_gt_pixel_weight = concat_img[6]

        if self.input_type == "raw_aug_seg":
            input_img = torch.stack(
                [augmented_raw_img, augmented_raw_transformed_img, augmented_scaled_seg_mask], dim=0
            )
        elif self.input_type == "raw_aug_duplicate":
            input_img = torch.stack(
                [augmented_raw_transformed_img, augmented_raw_transformed_img, augmented_raw_transformed_img], dim=0
            )
        elif self.input_type == "edt_v0":
            # TODO edt transform already done before the transform
            # augmented_scaled_seg_mask = scipy.ndimage.distance_transform_edt(augmented_scaled_seg_mask)
            augmented_scaled_seg_mask = normalize_edt(augmented_scaled_seg_mask, edt_max=4)
            input_img = torch.stack(
                [augmented_raw_transformed_img, augmented_raw_transformed_img, augmented_scaled_seg_mask], dim=0
            )
        elif self.input_type == "raw_duplicate":
            input_img = torch.stack([augmented_raw_img, augmented_raw_img, augmented_raw_img], dim=0)
        else:
            raise NotImplementedError

        if self.exclude_raw_input_bg:
            # Note that the augmented transformed image has all -1 values for pixels outside the cell
            # bg_mask = augmented_raw_img <= 0
            bg_mask = augmented_raw_transformed_img <= 0
            input_img[:, bg_mask] = 0

        input_img = input_img.float()

        aug_diff_img = concat_img[4, :, :]
        aug_diff_overseg = aug_diff_img < 0
        aug_diff_underseg = aug_diff_img > 0

        gt_mask = concat_img[3, :, :]
        gt_mask[gt_mask > 0.5] = 1
        gt_mask[gt_mask <= 0.5] = 0
        gt_binary = gt_mask

        # Deprecated and wrong gt mask edt below if resize
        if self.force_no_edt_aug:
            # Calculate gt_mask_edt based on augmented gt label mask
            gt_mask_edt = label_mask_to_edt_mask(augmented_gt_label_mask.cpu().numpy(), bg_val=self.bg_val)
            gt_mask_edt = torch.tensor(gt_mask_edt).float()
        else:
            # Use the augmented version of GT. Whether it is normed depends on train norm passed in.
            gt_mask_edt = concat_img[7, :, :]

        # apply edt to each label in gt label mask, and normalize edt to [0, 1]
        if self.apply_gt_seg_edt:
            combined_gt = torch.stack([gt_mask_edt, aug_diff_overseg, aug_diff_underseg], dim=0).float()
        else:
            combined_gt = torch.stack([gt_mask, aug_diff_overseg, aug_diff_underseg], dim=0).float()

        # Prepare ou_aux tensor: 4 classes auxillary output
        ou_aux = torch.tensor([0, 0, 0, 0]).float()
        if ou_aux_label is not None:
            if ou_aux_label == "overseg":
                ou_aux = torch.tensor([1, 0, 0, 0]).float()
            elif ou_aux_label == "underseg":
                ou_aux = torch.tensor([0, 1, 0, 0]).float()
            elif ou_aux_label == "dropout":
                ou_aux = torch.tensor([0, 0, 1, 0]).float()
            elif ou_aux_label == "correct":
                ou_aux = torch.tensor([0, 0, 0, 1]).float()
            else:
                raise ValueError("Unknown ou_aux value:", ou_aux_label)

        res = {
            "input": input_img,
            # "raw_img": augmented_raw_img,
            # "raw_transformed": augmented_raw_transformed_img,
            "seg_mask": augmented_scaled_seg_mask,  # If edt, this is the edt version
            "gt_mask_binary": gt_binary,
            "gt_mask": combined_gt,
            "idx": idx,
            "gt_label_mask": augmented_gt_label_mask,
            "ou_aux": ou_aux,
            "gt_pixel_weight": augmented_gt_pixel_weight,
        }
        if self.apply_gt_seg_edt:
            res["gt_mask_edt"] = gt_mask_edt
        return res

    def get_paths(self, idx):
        return {
            "raw_img": self.raw_img_paths[idx],
            "scaled_seg_mask": self.scaled_seg_mask_paths[idx],
            "gt_mask": self.gt_mask_paths[idx],
            "raw_seg": self.raw_seg_paths[idx],
            "raw_transformed_img": self.raw_transformed_img_paths[idx] if self.raw_transformed_img_paths else None,
            "aug_diff_img": self.aug_diff_img_paths[idx] if self.aug_diff_img_paths else None,
        }

    def get_gt_label_mask(self, idx) -> np.array:
        return np.array(Image.open(self.gt_label_mask_paths[idx]))

    def __len__(self):
        return len(self.raw_img_paths)


# class MultiChannelImageDataset(torch.utils.data.Dataset):
