import math
import pickle
import torch
import numpy as np


from typing import List, Union
import torch
from tqdm import tqdm
from livecellx.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from pathlib import Path
from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset
from livecellx.preprocess.utils import normalize_img_by_bitdepth


class SingleCellVaeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scs: List[SingleCellStatic],
        padding: int = 30,
        img_shape: tuple = (256, 256),
        transform=None,
        download=None,
        split="",
        cache_items: bool = True,
    ):
        self.scs = scs
        self.img_shape = img_shape
        self.img_size = np.prod(self.img_shape).astype(int)
        self.padding = padding
        self.transform = transform
        self.download = download
        self.split = split
        self.cache_items = cache_items
        self.cached_items = {}

    def __getitem__(self, idx, img_only=True) -> Union[dict, list, torch.Tensor, tuple]:
        if self.cache_items and idx in self.cached_items:
            return self.cached_items[idx]
        img = self.scs[idx].get_img_crop(padding=self.padding)
        img = normalize_img_by_bitdepth(img, mean=127, bit_depth=8)
        img = img.reshape([1] + list(img.shape))
        img = torch.from_numpy(img).float()
        if self.transform:
            img = self.transform(img)
        # Resize images to the specified shape
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=self.img_shape, mode="bilinear", align_corners=False
        ).squeeze(0)
        if self.cache_items:
            self.cached_items[idx] = img
        if img_only:
            return img
        else:
            return {
                "image": img,
                "idx": torch.tensor(idx, dtype=torch.long),
            }

    def __len__(self):
        return len(self.scs)


def scs_train_test_dataloader(
    sctc_path=Path(f"./notebook_results/MCF10A_a549_combined/MCF10A_A549_combined_sctc.json"),
    scs=None,
    padding=30,
    img_shape=(256, 256),
    batch_size=100,
    preload_dataset_to_memory=True,
):
    # if path is string, convert to Path
    if isinstance(sctc_path, str):
        sctc_path = Path(sctc_path)

    if scs is None:
        print("[INFO] Loading SingleCellTrajectoryCollection from path: {}".format(sctc_path))
        assert sctc_path is not None, "sctc_path must be provided if scs is None."
        assert sctc_path.exists(), f"Path {sctc_path} does not exist."
        sctc = SingleCellTrajectoryCollection.load_from_json_file(sctc_path)
        all_scs = sctc.get_all_scs()
    else:
        all_scs = scs
    # Set seed, split data into train and test sets
    np.random.seed(77)
    np.random.shuffle(np.array(all_scs))
    split_idx = int(0.8 * len(all_scs))
    train_scs = all_scs[:split_idx]
    test_scs = all_scs[split_idx:]
    # Create datasets
    train_dataset = SingleCellVaeDataset(
        train_scs,
        padding=padding,
        img_shape=img_shape,
    )
    test_dataset = SingleCellVaeDataset(
        test_scs,
        padding=padding,
        img_shape=img_shape,
    )
    if preload_dataset_to_memory:
        # Load datasets into memory
        for idx in tqdm(range(len(train_dataset)), desc="Loading train dataset into memory"):
            train_dataset[idx]
        for idx in tqdm(range(len(test_dataset)), desc="Loading test dataset into memory"):
            test_dataset[idx]

        # Load again, test if items are cached
        for idx in tqdm(range(len(train_dataset)), desc="Testing train dataset cache"):
            item = train_dataset[idx]
        for idx in tqdm(range(len(test_dataset)), desc="Testing test dataset cache"):
            item = test_dataset[idx]

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=32,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=32,
    )
    return train_loader, test_loader
