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
from livecellx.core.single_cell import SingleCellStatic
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
        elif subdirs is None:
            # Constrcut subdirs from raw_img_paths
            self.subdirs = [str(Path(path).parent.name) for path in self.raw_img_paths]
            self.subdirs = pd.Series(self.subdirs)
        else:
            self.subdirs = subdirs
        assert self.subdirs is not None, "subdirs of samples must be provided."

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

    def get_raw_seg(self, idx) -> np.ndarray:
        return np.array(Image.open(self.raw_seg_paths[idx]))

    def get_scale(self, idx):
        return self.scales[idx]

    def get_subdir(self, idx):
        if isinstance(self.subdirs, pd.Series):
            return self.subdirs.iloc[idx]
        return self.subdirs[idx]

    @staticmethod
    def label_mask_to_edt(label_mask: np.ndarray, bg_val=0):
        label_mask = label_mask.astype(np.uint8)
        labels = set(np.unique(label_mask))
        labels.remove(bg_val)
        res_edt = np.zeros_like(label_mask)
        for label in labels:
            tmp_bin_mask = label_mask == label
            tmp_edt = scipy.ndimage.morphology.distance_transform_edt(tmp_bin_mask)
            res_edt = np.maximum(res_edt, tmp_edt)
        return res_edt

    def _load_raw_data(self, idx: int):
        """
        Load raw images, masks, and auxiliary data from disk for a given index,
        without applying any transformations or conversions beyond reading from disk.

        Returns:
            dict:
                - "augmented_raw_img": PIL Image
                - "scaled_seg_mask": PIL Image
                - "gt_mask": PIL Image
                - "augmented_raw_transformed_img": PIL Image
                - "aug_diff_img": PIL Image
                - "gt_label_mask": numpy array
                - "ou_aux_label": str or None
                - "gt_pixel_weight": numpy array (if use_gt_pixel_weight=True), else None
        """
        # Load images directly from disk
        augmented_raw_img = Image.open(self.raw_img_paths[idx])
        scaled_seg_mask = Image.open(self.scaled_seg_mask_paths[idx])
        gt_mask = Image.open(self.gt_mask_paths[idx])
        augmented_raw_transformed_img = Image.open(self.raw_transformed_img_paths[idx])
        aug_diff_img = Image.open(self.aug_diff_img_paths[idx])
        gt_label_mask__np = np.array(Image.open(self.gt_label_mask_paths[idx]))

        ou_aux_label = self.raw_df["ou_aux"].iloc[idx] if "ou_aux" in self.raw_df.columns else None

        # Load pixel weights if needed
        if self.use_gt_pixel_weight:
            gt_pixel_weight = np.load(self.gt_pixel_weight_paths[idx])
        else:
            gt_pixel_weight = np.ones_like(gt_label_mask__np)

        return {
            "augmented_raw_img": augmented_raw_img,
            "scaled_seg_mask": scaled_seg_mask,
            "gt_mask": gt_mask,
            "augmented_raw_transformed_img": augmented_raw_transformed_img,
            "aug_diff_img": aug_diff_img,
            "gt_label_mask": gt_label_mask__np,
            "ou_aux_label": ou_aux_label,
            "gt_pixel_weight": gt_pixel_weight,
        }

    @staticmethod
    def _prepare_sc_inference_data(
        sc: SingleCellStatic,
        padding_pixels: int = 0,
        dtype=float,
        remove_bg=True,
        one_object=True,
        scale=0,
        bbox=None,
        normalize_crop=True,
    ):
        # TODO

        raw_img_crop = sc.get_img_crop(bbox=bbox, padding=padding_pixels)
        seg_crop = sc.get_mask_crop(bbox=bbox, padding=padding_pixels)
        if normalize_crop:
            raw_img_crop = normalize_img_to_uint8(raw_img_crop)
        raw_transformed_img = raw_img_crop.copy().astype(dtype)
        raw_transformed_img[(seg_crop.astype(int) < 1)] *= -1.0
        raw_transformed_img = raw_transformed_img.astype(dtype)

        # Normalize the images
        # raw_transformed_img = normalize_img_to_uint8(raw_transformed_img)

        raw_data = {
            "augmented_raw_img": raw_img_crop,
            "scaled_seg_mask": seg_crop,
            "gt_mask": np.zeros_like(raw_img_crop),
            "augmented_raw_transformed_img": raw_transformed_img,
            "aug_diff_img": np.zeros_like(raw_img_crop),
            "gt_label_mask": np.zeros_like(raw_img_crop),
            "ou_aux_label": None,
            "gt_pixel_weight": np.ones_like(raw_img_crop),
        }
        return raw_data

    @staticmethod
    def prepare_and_augment_data(
        raw_data: dict,
        input_type: str,
        bg_val: float,
        normalize_uint8: bool,
        exclude_raw_input_bg: bool,
        force_no_edt_aug: bool,
        apply_gt_seg_edt: bool,
        transform=None,
    ):
        """
        Receive the loaded data (in PIL/np form), convert to tensors, and apply
        necessary transformations and augmentations including normalization, EDT
        transforms, and final stacking.

        Args:
            raw_data (dict): Output from a data loading function.
            idx (int): Index of the item.
            input_type (str): Type of input formatting (e.g. "raw_aug_seg", "edt_v0").
            bg_val (float): Background value used in EDT calculations.
            normalize_uint8 (bool): Whether to normalize images to uint8 range.
            exclude_raw_input_bg (bool): Whether to zero out the input where the raw transformed image is background.
            force_no_edt_aug (bool): If True, re-compute EDT from augmented gt_label_mask after transforms.
            apply_gt_seg_edt (bool): If True, the GT mask will contain EDT as the first channel.
            transform (callable, optional): Augmentation transform to be applied on concatenated tensor.

        Returns:
            dict: Contains all tensors ready for model input:
                - "input": The input image tensor.
                - "seg_mask": The segmented mask tensor.
                - "gt_mask_binary": The binary ground truth mask tensor.
                - "gt_mask": The combined ground truth tensor.
                - "gt_label_mask": The ground truth label mask tensor.
                - "ou_aux": The auxiliary output tensor.
                - "gt_pixel_weight": The ground truth pixel weight tensor.
                - "gt_mask_edt" (optional): The EDT-transformed ground truth mask tensor if apply_gt_seg_edt is True.
        """
        # Extract raw data
        augmented_raw_img = raw_data["augmented_raw_img"]
        scaled_seg_mask = raw_data["scaled_seg_mask"]
        gt_mask = raw_data["gt_mask"]
        augmented_raw_transformed_img = raw_data["augmented_raw_transformed_img"]
        aug_diff_img = raw_data["aug_diff_img"]
        gt_label_mask__np = raw_data["gt_label_mask"]
        ou_aux_label = raw_data["ou_aux_label"]
        gt_pixel_weight__np = raw_data["gt_pixel_weight"]

        # Normalize if needed
        if normalize_uint8:
            augmented_raw_img = normalize_img_to_uint8(np.array(augmented_raw_img))
            augmented_raw_transformed_img = normalize_img_to_uint8(np.array(augmented_raw_transformed_img))
        else:
            augmented_raw_img = np.array(augmented_raw_img)
            augmented_raw_transformed_img = np.array(augmented_raw_transformed_img)

        # Convert all to tensors
        augmented_raw_img = torch.tensor(augmented_raw_img).float()
        scaled_seg_mask = torch.tensor(np.array(scaled_seg_mask)).float()
        gt_mask = torch.tensor(np.array(gt_mask)).long()
        augmented_raw_transformed_img = torch.tensor(augmented_raw_transformed_img).float()
        aug_diff_img = torch.tensor(np.array(aug_diff_img)).float()
        gt_label_mask = torch.tensor(gt_label_mask__np).long()
        gt_pixel_weight = torch.tensor(gt_pixel_weight__np).float()

        # Apply EDT for certain input types
        if input_type in ["edt_v0", "edt_v1"]:
            scaled_seg_mask_edt = label_mask_to_edt_mask(scaled_seg_mask, bg_val=bg_val)
            scaled_seg_mask = torch.tensor(scaled_seg_mask_edt).float()

        gt_label_edt = torch.tensor(label_mask_to_edt_mask(gt_label_mask, bg_val=bg_val)).float()

        # Stack for joint augmentation
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

        # Apply transform (augmentation)
        if transform:
            concat_img = transform(concat_img)

        # Unpack augmented results
        augmented_raw_img = concat_img[0]
        augmented_raw_transformed_img = concat_img[1]
        augmented_scaled_seg_mask = concat_img[2]
        gt_mask = concat_img[3]
        aug_diff_img = concat_img[4]
        augmented_gt_label_mask = concat_img[5].long()
        augmented_gt_pixel_weight = concat_img[6]
        gt_label_edt = concat_img[7]

        # Prepare the input based on input_type
        if input_type == "raw_aug_seg":
            input_img = torch.stack(
                [
                    augmented_raw_img,
                    augmented_raw_transformed_img,
                    augmented_scaled_seg_mask,
                ],
                dim=0,
            )
        elif input_type == "raw_aug_duplicate":
            input_img = torch.stack(
                [
                    augmented_raw_transformed_img,
                    augmented_raw_transformed_img,
                    augmented_raw_transformed_img,
                ],
                dim=0,
            )
        elif input_type == "edt_v0":
            if augmented_scaled_seg_mask.is_cuda:
                augmented_scaled_seg_mask = augmented_scaled_seg_mask.cpu()
            edt_np = augmented_scaled_seg_mask.numpy()
            edt_np = normalize_edt(edt_np, edt_max=5)
            edt_t = torch.tensor(edt_np)
            input_img = torch.stack(
                [augmented_raw_transformed_img, augmented_raw_transformed_img, edt_t],
                dim=0,
            )
        elif input_type == "edt_v1":
            if augmented_scaled_seg_mask.is_cuda:
                augmented_scaled_seg_mask = augmented_scaled_seg_mask.cpu()
            edt_np = augmented_scaled_seg_mask.numpy()
            edt_np = normalize_edt(edt_np, edt_max=5)
            edt_t = torch.tensor(edt_np)
            input_img = torch.stack([augmented_raw_img, edt_t, torch.zeros_like(edt_t)], dim=0)
        elif input_type == "raw_duplicate":
            input_img = torch.stack([augmented_raw_img, augmented_raw_img, augmented_raw_img], dim=0)
        else:
            raise NotImplementedError(f"Unknown input_type: {input_type}")

        # Exclude background if needed
        if exclude_raw_input_bg:
            bg_mask = augmented_raw_transformed_img <= 0
            input_img[:, bg_mask] = 0

        input_img = input_img.float()

        # Process GT masks
        gt_mask[gt_mask > 0.5] = 1
        gt_mask[gt_mask <= 0.5] = 0
        gt_binary = gt_mask

        # Handle EDT if forced without augmentation
        if force_no_edt_aug:
            # Recompute from augmented gt_label_mask
            gt_mask_edt_np = label_mask_to_edt_mask(augmented_gt_label_mask.cpu().numpy(), bg_val=bg_val)
            gt_mask_edt = torch.tensor(gt_mask_edt_np).float()
        else:
            gt_mask_edt = gt_label_edt

        aug_diff_overseg = aug_diff_img < 0
        aug_diff_underseg = aug_diff_img > 0

        # Combine GT
        if apply_gt_seg_edt:
            combined_gt = torch.stack([gt_mask_edt, aug_diff_overseg, aug_diff_underseg], dim=0).float()
        else:
            combined_gt = torch.stack([gt_mask, aug_diff_overseg, aug_diff_underseg], dim=0).float()

        # Prepare ou_aux
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

        result = {
            "input": input_img,
            "seg_mask": augmented_scaled_seg_mask,
            "gt_mask_binary": gt_binary,
            "gt_mask": combined_gt,
            "gt_label_mask": augmented_gt_label_mask,
            "ou_aux": ou_aux,
            "gt_pixel_weight": augmented_gt_pixel_weight,
        }

        if apply_gt_seg_edt:
            result["gt_mask_edt"] = gt_mask_edt

        return result

    def __getitem__(self, idx: int):
        # Step 1: Load raw data (no transforms)
        raw_data = self._load_raw_data(idx)

        # Step 2: Prepare and augment data
        result = self.prepare_and_augment_data(
            raw_data,
            self.input_type,
            self.bg_val,
            self.normalize_uint8,
            self.exclude_raw_input_bg,
            self.force_no_edt_aug,
            self.apply_gt_seg_edt,
            self.transform,
        )

        # Attach the index
        result["idx"] = idx

        return result

    def get_paths(self, idx):
        return {
            "raw_img": self.raw_img_paths[idx],
            "scaled_seg_mask": self.scaled_seg_mask_paths[idx],
            "gt_mask": self.gt_mask_paths[idx],
            "raw_seg": self.raw_seg_paths[idx],
            "raw_transformed_img": (self.raw_transformed_img_paths[idx] if self.raw_transformed_img_paths else None),
            "aug_diff_img": (self.aug_diff_img_paths[idx] if self.aug_diff_img_paths else None),
        }

    def get_gt_label_mask(self, idx) -> np.ndarray:
        return np.array(Image.open(self.gt_label_mask_paths[idx]))

    def __len__(self):
        return len(self.raw_img_paths)


# class MultiChannelImageDataset(torch.utils.data.Dataset):
