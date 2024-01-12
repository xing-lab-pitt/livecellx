from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning.loggers import TensorBoardLogger


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None, data_dir=None):
        self.dataframe = dataframe
        self.transform = transform

        assert data_dir is not None
        self.data_dir = data_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_relative_path = self.dataframe.iloc[idx]["img_path"]
        img_path = self.data_dir / img_relative_path
        image = Image.open(img_path)
        label = int(self.dataframe.iloc[idx]["label_index"])

        if self.transform:
            image = self.transform(image)

        return image, label


class DataModule(pl.LightningDataModule):
    def __init__(self, train_df, valid_df, batch_size=32):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.batch_size = batch_size

    def setup(self, stage=None):
        # transform defined here
        self.train_dataset = CustomDataset(self.train_df, transform=transform)
        self.valid_dataset = CustomDataset(self.valid_df, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=32)
