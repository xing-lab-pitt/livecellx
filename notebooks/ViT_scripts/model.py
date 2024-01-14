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
import argparse
from dataset import CustomDataset, DataModule


class LcaImageClassificationModel(pl.LightningModule):
    """LivecellAction Image Classification Model"""

    def __init__(self, model="vit_b_16"):
        super().__init__()
        if model=="vit_b_16":
            self.model = models.vit_b_16(pretrained=True)
        elif model=="resnet50":
            self.model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"model {model} not recognized")
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        correct = torch.sum(preds == labels).item()
        accuracy = correct / len(labels)

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        correct = torch.sum(preds == labels).item()
        accuracy = correct / len(labels)

        # Logging
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer
