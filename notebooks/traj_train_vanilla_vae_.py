# %%
from livecell_tracker.external import torch_vae

# %%
from livecell_tracker.external.torch_vae.models.vanilla_vae import VanillaVAE
import numpy as np


# %% [markdown]
# Train

# %%
from typing import List, Union
import torch
from livecell_tracker.core import SingleCellStatic
from pathlib import Path
from livecell_tracker.core.datasets import LiveCellImageDataset, SingleImageDataset
from livecell_tracker.preprocess.utils import normalize_img_to_uint8


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

    def __getitem__(self, idx) -> Union[dict, list, torch.Tensor, tuple]:
        if self.cache_items and idx in self.cached_items:
            return self.cached_items[idx]
        img = self.scs[idx].get_img_crop(padding=self.padding)
        img = normalize_img_to_uint8(img)
        img = img.reshape([1] + list(img.shape))
        img = torch.from_numpy(img).float()
        if self.transforms:
            img = self.transforms(img)
        # return {
        #     "input": img,
        #     "img": img,
        # }
        #
        if self.cache_items:
            self.cached_items[idx] = img
        return img, torch.tensor([])

    def __len__(self):
        return len(self.scs)


# %%
# dataset_dir_path = Path(
#     "../datasets/test_data_STAV-A549/DIC_data"
# )
# mask_dataset_path = Path("../datasets/test_data_STAV-A549/mask_data")
# mask_dataset = LiveCellImageDataset(mask_dataset_path, ext="png")
# dic_dataset = LiveCellImageDataset(dataset_dir_path, ext="tif")

dataset_dir_path = Path("../datasets/EBSS_Starvation/tif_STAV-A549_VIM_24hours_NoTreat_NA_YL_Ti2e_2022-12-21/XY16/")

mask_dataset_path = Path(
    "../datasets/EBSS_Starvation/tif_STAV-A549_VIM_24hours_NoTreat_NA_YL_Ti2e_2022-12-21/out/XY16/seg"
)
mask_dataset = LiveCellImageDataset(mask_dataset_path, ext="png")
import glob

time2url = sorted(glob.glob(str((Path(dataset_dir_path) / Path("*_DIC.tif")))))
time2url = {i: path for i, path in enumerate(time2url)}
dic_dataset = LiveCellImageDataset(time2url=time2url, ext="tif")

# %%
from skimage.measure import regionprops
from livecell_tracker.segment.utils import prep_scs_from_mask_dataset

single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)
for sc in single_cells:
    sc.cache = True

print("total number of single cells: ", len(single_cells))


# %%
from typing import Optional, Sequence, Union
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class ScVAEDataset(LightningDataModule):
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

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(148),
                transforms.RandomCrop(self.patch_size, pad_if_needed=True)
                # transforms.Resize(self.patch_size),
                # transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(148),
                transforms.RandomCrop(self.patch_size, pad_if_needed=True)
                # transforms.Resize(self.patch_size),
                # transforms.ToTensor(),
            ]
        )

        self.train_dataset = SingleCellVaeDataset(
            self.train_scs,
            split="train",
            transforms=train_transforms,
            download=False,
            img_shape=self.patch_size,
        )

        # Replace CelebA with your dataset
        self.val_dataset = SingleCellVaeDataset(
            self.val_scs,
            split="test",
            transforms=val_transforms,
            download=False,
            img_shape=self.patch_size,
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


# %%
import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from livecell_tracker.external.torch_vae.models import *
from livecell_tracker.external.torch_vae.experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from livecell_tracker.external.torch_vae.dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin


config = {
    "model_params": {"name": "VanillaVAE", "in_channels": 1, "latent_dim": 128},
    "data_params": {
        "data_path": "Data/",
        "train_batch_size": 256,
        "val_batch_size": 256,
        "patch_size": 64,
        "num_workers": 0,
    },
    "exp_params": {
        "LR": 0.00001,
        "weight_decay": 0.0,
        # "scheduler_gamma": 0.95,
        "kld_weight": 0.00025,
        "manual_seed": 1265,
    },
    "trainer_params": {
        "gpus": 1,
        "max_epochs": 20000,
    },
    "logging_params": {"save_dir": "vae_logs/", "name": "VanillaVae"},
}

tb_logger = TensorBoardLogger(
    save_dir=config["logging_params"]["save_dir"],
    name=config["model_params"]["name"],
)

# For reproducibility
seed_everything(config["exp_params"]["manual_seed"], True)

# model = vae_models[config['model_params']['name']](**config['model_params'])
vae_img_shape = (256, 256)

in_channels, latent_dim = 1, config["model_params"]["latent_dim"]
vae_model = VanillaVAE(in_channels, latent_dim)
experiment = VAEXperiment(vae_model, config["exp_params"]).cuda()


# data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data = ScVAEDataset(
    single_cells, single_cells, **config["data_params"], pin_memory=config["trainer_params"]["gpus"] != 0
)
# data.setup()
runner = Trainer(
    logger=tb_logger,
    callbacks=[
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=2, dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), monitor="val_loss", save_last=True
        ),
    ],
    #  strategy=DDPPlugin(find_unused_parameters=False),
    **config["trainer_params"],
)


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
