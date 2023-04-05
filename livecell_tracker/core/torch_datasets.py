from typing import List, Union
import torch
from pathlib import Path
from skimage.measure import regionprops
from typing import Optional, Sequence, Union
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from functools import partial

from livecell_tracker.core import SingleCellStatic
from livecell_tracker.core.datasets import LiveCellImageDataset, SingleImageDataset
from livecell_tracker.preprocess.utils import normalize_img_to_uint8
from livecell_tracker.segment.utils import prep_scs_from_mask_dataset


class SingleCellVaeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scs: List[SingleCellStatic],
        padding: int = 0,
        img_shape: tuple = (256, 256),
        transforms=None,
        download=None,
        split="",
        cache_items=True,
        img_only=False,
    ):
        self.scs = scs
        self.img_shape = img_shape
        self.img_size = np.prod(self.img_shape).astype(int)
        self.padding = padding
        self.transforms = transforms
        self.download = download
        self.split = split
        self.cache_items = cache_items
        self.cached_items = {}
        self.img_only = img_only

    def __getitem__(self, idx) -> Union[dict, list, torch.Tensor, tuple]:
        if self.cache_items and idx in self.cached_items:
            return self.cached_items[idx]
        img = self.scs[idx].get_img_crop(
            padding=self.padding, preprocess_img_func=partial(normalize_img_to_uint8, dtype=float)
        )

        # normalize on the single cell level
        # img = self.scs[idx].get_img_crop(padding=self.padding)
        # img = normalize_img_to_uint8(img)

        img = img.reshape([1] + list(img.shape))
        ####### debug ########
        # from matplotlib import pyplot as plt
        # plt.imshow(img[0])
        # plt.savefig("./sample_test_img.png")
        # plt.clf()
        # plt.hist(img[0].flatten(), bins=100)
        # plt.savefig("./sample_test_img_hist.png")
        ####### end debug ########
        img = torch.from_numpy(img).float()
        if self.transforms:
            img = self.transforms(img)

        # plt.hist(img[0].cpu().numpy().flatten(), bins=100)
        # plt.savefig("./sample_test_img_hist_transformed.png")
        # exit(0)

        # return {
        #     "input": img,
        #     "img": img,
        # }
        #
        if self.cache_items:
            self.cached_items[idx] = (img, torch.tensor([]))

        if self.img_only:
            return img
        return img, torch.tensor([])

    def __len__(self):
        return len(self.scs)


class ScVaeDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        train_scs: List[SingleCellStatic],
        val_scs: List[SingleCellStatic],
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        img_only=False,
        **kwargs,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_scs = train_scs
        self.val_scs = val_scs
        self.img_only = img_only

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(148),
                # transforms.RandomCrop(self.patch_size, pad_if_needed=True)
                transforms.Resize((self.patch_size, self.patch_size)),
                # transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(148),
                # transforms.RandomCrop(self.patch_size, pad_if_needed=True)
                transforms.Resize((self.patch_size, self.patch_size)),
                # transforms.ToTensor(),
            ]
        )

        self.train_dataset = SingleCellVaeDataset(
            self.train_scs,
            split="train",
            transforms=train_transforms,
            download=False,
            img_shape=self.patch_size,
            img_only=self.img_only,
        )

        # Replace CelebA with your dataset
        self.val_dataset = SingleCellVaeDataset(
            self.val_scs,
            split="test",
            transforms=val_transforms,
            download=False,
            img_shape=self.patch_size,
            img_only=self.img_only,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
