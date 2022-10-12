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


class LiveCellImageDataset(torch.utils.data.Dataset):
    """Dataset that reads in various features"""

    def __init__(
        self, dir_path=None, ext="tif", max_cache_size=50, name="livecell-base", num_imgs=None, force_posix_path=True
    ):

        if isinstance(dir_path, str):
            # dir_path = Path(dir_path)
            dir_path = PurePosixPath(dir_path)
        elif isinstance(dir_path, Path) and force_posix_path:
            dir_path = PurePosixPath(dir_path)

        self.data_dir_path = dir_path
        self.ext = ext
        self.update_img_paths()

        # force posix path
        if force_posix_path:
            # TODO: fix pathlib issues on windows;
            # TODO should work without .replace('\\', '/'), but it doesn't on Ke's windows py3.8; need confirmation
            self.img_path_list = [str(PurePosixPath(path)).replace("\\", "/") for path in list(self.img_path_list)]
        if num_imgs is not None:
            self.img_path_list = self.img_path_list[:num_imgs]
        self.img_idx2img = {}
        self.max_cache_size = max_cache_size
        self.img_idx_queue = deque()
        self.name = name

    def update_img_paths(self):
        if self.data_dir_path is None:
            self.img_path_list = []
            return
        assert self.ext, "ext must be specified"
        self.img_path_list = sorted(glob.glob(str((Path(self.data_dir_path) / Path("*.%s" % (self.ext))))))
        # print("%d %s img file paths loaded: " % (len(self.img_path_list), self.ext))

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

    def get_dataset_name(self):
        return self.name

    def get_dataset_path(self):
        return self.data_dir_path

    def __getitem__(self, idx):
        if idx in self.img_idx2img:
            return self.img_idx2img[idx]
        img = Image.open(self.img_path_list[idx])
        img = np.array(img)
        self.insert_cache(img, idx)
        return img

    def to_json_dict(self) -> dict:
        # img_path_list = [str(PurePosixPath(path)) for path in self.img_path_list]
        return {
            "name": self.name,
            "data_dir_path": str(self.data_dir_path),
            "img_path_list": self.img_path_list,
            "max_cache_size": int(self.max_cache_size),
            "ext": self.ext,
        }

    def write_json(self, path=None):
        if path is None:
            return json.dumps(self.to_dict())
        else:
            with open(path, "w+") as f:
                json.dump(self.to_dict(), f)

    def load_from_json_dict(self, json_dict):
        self.name = json_dict["name"]
        self.data_dir_path = json_dict["data_dir_path"]
        self.update_img_paths()

        self.img_path_list = json_dict["img_path_list"]
        self.max_cache_size = json_dict["max_cache_size"]
        self.ext = json_dict["ext"]
        return self
