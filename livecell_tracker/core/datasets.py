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
from typing import Callable, List, Dict, Union

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


def read_img_default(url: str) -> np.ndarray:
    img = Image.open(url)
    img = np.array(img)
    return img


class LiveCellImageDataset(torch.utils.data.Dataset):
    """Dataset for loading images into RAM, possibly cache images and load them on demand.
    This class only contains one channel's imaging data. For multichannel data, we assume you have a single image for each channel.
    For the case where your images are stored in a single file, #TODO: you can use the MultiChannelImageDataset class.
    """

    def __init__(
        self,
        dir_path=None,
        time2url: Dict[int, Union[str, Path]] = None,
        name="livecell-base",
        ext="tif",
        max_cache_size=50,
        num_imgs=None,
        force_posix_path=True,
        read_img_url_func: Callable = read_img_default,
        index_by_time=True,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        dir_path : _type_, optional
            _description_, by default None
        time2url : Dict[int, str], optional
            _description_, by default None
        name : str, optional
            _description_, by default "livecell-base"
        ext : str, optional
            _description_, by default "tif"
        max_cache_size : int, optional
            _description_, by default 50
        num_imgs : _type_, optional
            _description_, by default None
        force_posix_path : bool, optional
            _description_, by default True
        read_img_url_func : Callable, optional
            _description_, by default read_img_default
        index_by_time : bool, optional
            _description_, by default True
        """

        self.read_img_url_func = read_img_url_func
        self.index_by_time = index_by_time

        # force posix path if dir_path is passed in
        if isinstance(dir_path, str):
            # dir_path = Path(dir_path)
            dir_path = PurePosixPath(dir_path)
        elif isinstance(dir_path, Path) and force_posix_path:
            dir_path = PurePosixPath(dir_path)

        self.data_dir_path = dir_path
        self.ext = ext

        if time2url is None:
            self.update_time2url_from_dir_path()

        elif isinstance(time2url, list):
            self.time2url = {i: path for i, path in enumerate(time2url)}
        else:
            self.time2url = time2url

        # force posix path
        if force_posix_path:
            # TODO: fix pathlib issues on windows;
            # TODO should work without .replace('\\', '/'), but it doesn't on Ke's windows py3.8; need confirmation
            self.time2url = {
                time: str(Path(path).as_posix()).replace("\\", "/") for time, path in self.time2url.items()
            }

        if num_imgs is not None:
            tmp_tuples = list(self.time2url.items())
            tmp_tuples = sorted(tmp_tuples, key=lambda x: x[0])
            tmp_tuples = tmp_tuples[:num_imgs]
            self.time2url = {time: path for time, path in tmp_tuples}
        self.times = list(self.time2url.keys())

        self.cache_img_idx_to_img = {}
        self.max_cache_size = max_cache_size
        self.img_idx_queue = deque()
        self.name = name

    def update_time2url_from_dir_path(self):
        """Update the time2url dictionary from the directory path"""
        if self.data_dir_path is None:
            self.time2url = {}
            return
        assert self.ext, "ext must be specified"
        self.time2url = sorted(glob.glob(str((Path(self.data_dir_path) / Path("*.%s" % (self.ext))))))
        self.time2url = {i: path for i, path in enumerate(self.time2url)}
        self.times = list(self.time2url.keys())
        print("%d %s img file paths loaded: " % (len(self.time2url), self.ext))
        return self.time2url

    def __len__(self):
        return len(self.time2url)

    def insert_cache(self, img, idx):
        self.cache_img_idx_to_img[idx] = img
        self.img_idx_queue.append(idx)

        # Do not move this block to the top of the function: corner case for max_cache_size = 0
        if len(self.cache_img_idx_to_img) > self.max_cache_size:
            pop_index = self.img_idx_queue.popleft()
            pop_img = self.cache_img_idx_to_img[pop_index]
            self.cache_img_idx_to_img.pop(pop_index)
            del pop_img

    # TODO: refactor path -> url
    def get_img_path(self, time) -> str:
        """Get the path of the image at some time

        Parameters
        ----------
        time : _type_
            _description_

        Returns
        -------
        str
            _description_
        """
        return self.time2url[time]

    def get_dataset_name(self):
        return self.name

    def get_dataset_path(self):

        return self.data_dir_path

    def __getitem__(self, idx) -> np.ndarray:
        if idx in self.cache_img_idx_to_img:
            return self.cache_img_idx_to_img[idx]
        # TODO: optimize
        if self.index_by_time:
            img = self.get_img_by_time(idx)
        else:
            img = self.get_img_by_idx(idx)
        return img

    def to_json_dict(self) -> dict:
        """Return the dataset info as a dictionary object"""
        # img_path_list = [str(PurePosixPath(path)) for path in self.img_path_list]
        return {
            "name": self.name,
            "data_dir_path": str(self.data_dir_path),
            "img_path_list": self.time2url,
            "max_cache_size": int(self.max_cache_size),
            "ext": self.ext,
        }

    # TODO: refactor
    def write_json(self, path=None):
        """Write the dataset info to a local json file."""
        if path is None:
            return json.dumps(self.to_dict())
        else:
            with open(path, "w+") as f:
                json.dump(self.to_dict(), f)

    def load_from_json_dict(self, json_dict, update_time2url_from_dir_path=False):
        """Load from a json dict. If update_img_paths is True, then we will update the img_path_list based on the data_dir_path.

        Parameters
        ----------
        json_dict : _type_
            _description_
        update_img_paths : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        self.name = json_dict["name"]
        self.data_dir_path = json_dict["data_dir_path"]
        self.ext = json_dict["ext"]
        if update_time2url_from_dir_path:
            self.update_time2url_from_dir_path()
        else:
            self.time2url = json_dict["img_path_list"]
        self.max_cache_size = json_dict["max_cache_size"]
        return self

    def to_dask(self, times=None, ram=False):
        """convert to a dask array for napari visualization"""
        import dask.array as da
        from dask import delayed

        if times is None:
            times = self.times
        if ram:
            return da.stack([da.from_array(self.time2url[time]) for time in times])

        lazy_reader = delayed(self.read_img_url_func)
        lazy_arrays = [lazy_reader(self.time2url[time]) for time in times]
        img_shape = self.infer_shape()
        dask_arrays = [da.from_delayed(lazy_array, shape=img_shape, dtype=int) for lazy_array in lazy_arrays]
        return da.stack(dask_arrays)

    def get_img_by_idx(self, idx):
        """Get an image by some index in the times list"""
        time = self.times[idx]
        img = self.read_img_url_func(self.time2url[time])
        return img

    def get_img_by_time(self, time) -> np.array:
        """Get an image by time"""
        return self.read_img_url_func(self.time2url[time])

    def get_img_by_url(self, url: str, substr=True, return_path_and_time=False, ignore_missing=False):
        """Get image by url

        Parameters
        ----------
        url : str
            _description_
        substr : bool, optional
            if true, match by substring. (url in _url or _url in url), by default True
        return_path_and_time : bool, optional
            if True return paths and time in the return values, , by default False
        ignore_missing : bool, optional
            ignore failure of matching and return None(s), by default False

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        found_url = None
        found_time = None

        def _cmp_equal(x, y):
            return x == y

        def _cmp_substr(x, y):
            return (x in y) or (y in x)

        cmp_func = _cmp_substr if substr else _cmp_equal

        for time, full_url in self.time2url.items():
            if (found_url is not None) and cmp_func(url, full_url):
                raise ValueError("Duplicate url found: %s" % url)
            if cmp_func(url, full_url):
                found_url = full_url
                found_time = time

        if found_url is None:
            if ignore_missing:
                return None, None, None if return_path_and_time else None
            else:
                raise ValueError("url not found: %s" % url)

        if return_path_and_time:
            return self.get_img_by_time(found_time), found_url, found_time
        return self.get_img_by_time(found_time)

    def infer_shape(self):
        """Infer the shape of the images in the dataset"""
        img = self.get_img_by_time(self.times[0])
        return img.shape

    def subset_by_time(self, min_time, max_time, prefix="_subset"):
        """Return a subset of the dataset based on time [min, max)"""
        times2url = {}
        for time in self.times:
            if time >= min_time and time < max_time:
                times2url[time] = self.time2url[time]
        return LiveCellImageDataset(
            time2url=times2url,
            name="_subset_" + self.name,
            ext=self.ext,
            read_img_url_func=self.read_img_url_func,
        )


class SingleImageDataset(LiveCellImageDataset):
    DEFAULT_TIME = 0

    def __init__(self, img, name=None, ext=".png"):
        super().__init__(
            time2url={SingleImageDataset.DEFAULT_TIME: "InMemory"},
            name=name,
            ext=ext,
            read_img_url_func=self.read_img_url_func,
        )
        self.img = img

    def read_img_url_func(self, url):
        return self.img.copy()


# TODO
# class MultiChannelImageDataset(torch.utils.data.Dataset):
