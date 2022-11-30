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


# class CorrectSegNetData(data.Dataset):
#     def __init__(self, livecell_dataset: LiveCellImageDataset, segnet_dataset: LiveCellImageDataset):
#         self.livecell_dataset = livecell_dataset
#         self.segnet_dataset = segnet_dataset


class CorrectSegNetDataset(torch.utils.data.Dataset):
    """Dataset for training CorrectSegNetDatasert"""

    def __init__(
        self,
        raw_img_paths: List[str],
        seg_mask_paths: List[str],
        gt_mask_paths: List[str],
        raw_seg_paths: List[str],
        scales: List[float],
        transform=None,
        raw_transformed_img_paths: List[str] = None,
        aug_diff_img_paths: List[str] = None,
        input_type="raw_aug_seg",
        apply_gt_seg_edt=False,
    ):
        self.raw_img_paths = raw_img_paths
        self.scaled_seg_mask_paths = seg_mask_paths
        self.gt_mask_paths = gt_mask_paths
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
        print("input type:", self.input_type)
        print("if apply_gt_seg_edt:", self.apply_gt_seg_edt)

    def get_raw_seg(self, idx) -> np.array:
        return np.array(Image.open(self.raw_seg_paths[idx]))

    def get_scale(self, idx):
        return self.scales[idx]

    def __getitem__(self, idx):
        augmented_raw_img = Image.open(self.raw_img_paths[idx])
        scaled_seg_mask = Image.open(self.scaled_seg_mask_paths[idx])
        gt_mask = Image.open(self.gt_mask_paths[idx])
        augmented_raw_transformed_img = Image.open(self.raw_transformed_img_paths[idx])
        aug_diff_img = Image.open(self.aug_diff_img_paths[idx])

        augmented_raw_img = torch.tensor(np.array(augmented_raw_img)).float()
        scaled_seg_mask = torch.tensor(np.array(scaled_seg_mask)).float()
        gt_mask = torch.tensor(np.array(gt_mask)).long()
        augmented_raw_transformed_img = torch.tensor(np.array(augmented_raw_transformed_img)).float()
        aug_diff_img = torch.tensor(np.array(aug_diff_img)).float()

        # prepare for augmentation
        concat_img = torch.stack(
            [augmented_raw_img, augmented_raw_transformed_img, scaled_seg_mask, gt_mask.float(), aug_diff_img], dim=0
        )
        if self.transform:
            concat_img = self.transform(concat_img)

        augmented_raw_img = concat_img[0]
        augmented_raw_transformed_img = concat_img[1]
        augmented_scaled_seg_mask = concat_img[2]

        if self.input_type == "raw_aug_seg":
            input_img = torch.stack(
                [augmented_raw_img, augmented_raw_transformed_img, augmented_scaled_seg_mask], dim=0
            )
        elif self.input_type == "raw_aug_duplicate":
            input_img = torch.stack(
                [augmented_raw_transformed_img, augmented_raw_transformed_img, augmented_raw_transformed_img], dim=0
            )
        elif self.input_type == "edt_v0":
            # TODO edt transform
            augmented_scaled_seg_mask = scipy.ndimage.distance_transform_edt(augmented_scaled_seg_mask)
            input_img = torch.stack(
                [augmented_raw_transformed_img, augmented_raw_transformed_img, augmented_scaled_seg_mask], dim=0
            )
        else:
            raise NotImplementedError
        input_img = input_img.float()

        gt_mask = concat_img[3, :, :]
        # TODO if use EDT or other gt, disable the following line
        gt_mask[gt_mask > 0.5] = 1
        gt_mask[gt_mask <= 0.5] = 0

        if self.apply_gt_seg_edt:
            gt_mask = torch.tensor(scipy.ndimage.distance_transform_edt(gt_mask[:, :]))

        aug_diff_img = concat_img[4, :, :]
        aug_diff_overseg = aug_diff_img < 0
        aug_diff_underseg = aug_diff_img > 0
        combined_gt = torch.stack([gt_mask, aug_diff_overseg, aug_diff_underseg], dim=0).float()

        return {
            "input": input_img,
            # "raw_img": augmented_raw_img,
            # "raw_derived": augmented_raw_transformed_img,
            # "seg_mask": augmented_scaled_seg_mask,
            # "gt_mask": gt_mask,
            "gt_mask": combined_gt,
        }

    def get_paths(self, idx):
        return {
            "raw_img": self.raw_img_paths[idx],
            "scaled_seg_mask": self.scaled_seg_mask_paths[idx],
            "gt_mask": self.gt_mask_paths[idx],
            "raw_seg": self.raw_seg_paths[idx],
            "raw_transformed_img": self.raw_transformed_img_paths[idx] if self.raw_transformed_img_paths else None,
            "aug_diff_img": self.aug_diff_img_paths[idx] if self.aug_diff_img_paths else None,
        }

    def __len__(self):
        return len(self.raw_img_paths)


# class MultiChannelImageDataset(torch.utils.data.Dataset):
