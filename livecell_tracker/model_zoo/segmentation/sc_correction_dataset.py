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
        transform=None,
    ):
        self.raw_img_paths = raw_img_paths
        self.seg_mask_paths = seg_mask_paths
        self.gt_mask_paths = gt_mask_paths
        self.transform = transform
        assert (
            len(self.raw_img_paths) == len(self.seg_mask_paths) == len(self.gt_mask_paths)
        ), "The number of images, segmentation masks and ground truth masks must be the same."

    def __getitem__(self, idx):
        raw_img = Image.open(self.raw_img_paths[idx])
        seg_mask = Image.open(self.seg_mask_paths[idx])
        gt_mask = Image.open(self.gt_mask_paths[idx])

        raw_img = torch.tensor(np.array(raw_img)).float()
        seg_mask = torch.tensor(np.array(seg_mask)).float()
        gt_mask = torch.tensor(np.array(gt_mask)[np.newaxis, :, :]).long()

        input_img = torch.stack([raw_img, seg_mask, seg_mask], dim=0)
        input_img = input_img.float()
        if self.transform:
            input_img = self.transform(input_img)
            gt_mask = self.transform(gt_mask)
            gt_mask = gt_mask.squeeze(0)  # remove the first dimension added for Resize
            gt_mask[gt_mask > 0.5] = 1
            gt_mask[gt_mask <= 0.5] = 0
        return {
            "input": input_img,
            # "raw_img": raw_img,
            # "seg_mask": seg_mask,
            "gt_mask": gt_mask,
        }

    def __len__(self):
        return len(self.raw_img_paths)


# class MultiChannelImageDataset(torch.utils.data.Dataset):
