import gzip
import torch
import numpy as np
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import argparse
import time
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from datetime import timedelta

import pandas as pd
import glob
import os.path
from pathlib import Path
from PIL import Image
from collections import deque


class LiveCellImageDataset(torch.utils.data.Dataset):
    """Dataset that reads in various features"""

    def __init__(self, dir_path, ext="tif", max_cache_size=50):
        if isinstance(dir_path, str):
            dir_path = Path(dir_path)

        self.img_path_list = sorted(glob.glob(str(dir_path / ("*.%s" % (ext)))))
        self.img_idx2img = {}
        self.max_cache_size = max_cache_size
        self.img_idx_queue = deque()
        print("%d %s img file paths loaded: " % (len(self.img_path_list), ext))

    def __len__(self):
        return len(self.img_path_list)

    def insert_cache(self, img, idx):
        if len(self.img_idx2img) >= self.max_cache_size:
            pop_index = self.img_idx_queue.popleft()
            pop_img = self.img_idx2img[pop_index]
            self.img_idx2img.pop(pop_index)
            del pop_img
        self.img_idx2img[idx] = img
        self.img_idx_queue.append(idx)

    def get_img_path(self, idx):
        return self.img_path_list[idx]

    def __getitem__(self, idx):
        if idx in self.img_idx2img:
            return self.img_idx2img[idx]
        img = Image.open(self.img_path_list[idx])
        img = np.array(img)
        self.insert_cache(img, idx)
        return img
