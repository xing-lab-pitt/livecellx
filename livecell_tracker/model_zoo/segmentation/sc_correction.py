import os
import argparse

import numpy as np

import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from torch import nn
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torch.autograd import Variable
from torch.utils import data
from torch.utils import data
from livecell_tracker.core.datasets import LiveCellImageDataset
from livecell_tracker.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset


class CorrectSegNet(LightningModule):
    def __init__(
        self,
        lr=1e-3,
        batch_size=5,
        class_weights=[],
        model_type=None,
        num_workers=16,
        train_input_paths=None,
    ):
        """_summary_

        Parameters
        ----------
        lr : _type_, optional
            _description_, by default 1e-3
        batch_size : int, optional
            _description_, by default 5
        class_weights : list, optional
            _description_, by default []
        model_type : _type_, optional
            _description_, by default None
        num_workers : int, optional
            _description_, by default 16
        train_input_paths : _type_, optional
            a list of (raw_path, seg_mask_path, gt_path), by default None
        """

        super().__init__()

        self.class_weights = torch.tensor(class_weights).cuda()
        self.model_type = model_type
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DeepLabV3_ResNet50_Weights.DEFAULT")
        self.model.classifier[4] = nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))

        self.loss_func = nn.CrossEntropyLoss(weight=self.class_weights)
        self.learning_rate = lr
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(42)

        self.dims = (1, 412, 412)
        self.num_workers = num_workers
        self.train_input_paths = train_input_paths

        self.save_hyperparameters()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        x = x["out"]
        return nn.functional.softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch["input"], batch["gt_mask"]
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["input"], batch["gt_mask"]
        logits = self(x)
        loss = self.loss_func(logits, y)
        self.val_accuracy.update(logits, y.long())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        x, y = batch["input"], batch["gt_mask"]

        logits = self(x)

        loss = self.loss_func(logits, y)
        self.test_accuracy.update(logits, y.long())

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage=None):
        ################### datasets settings ###################
        img_paths, mask_paths, gt_paths = list(zip(*self.train_input_paths))
        self.full_dataset = CorrectSegNetDataset(img_paths, mask_paths, gt_paths)
        num_train_samples = int(len(self.full_dataset) * 0.7)
        num_val_samples = int((len(self.full_dataset) - num_train_samples) / 2)
        num_test_samples = len(self.full_dataset) - num_train_samples - num_val_samples

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [num_train_samples, num_val_samples, num_test_samples]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=self.generator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            generator=self.generator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            generator=self.generator,
        )

    def predict_step(self, batch):
        x, y = batch["input"], batch["gt_mask"]
        img_file_path, mask_file_path, img_idx = (
            batch["img_file_path"],
            batch["mask_file_path"],
            batch["index"],
        )

        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return [x, np.argmax(y, axis=1), preds], [
            img_file_path,
            mask_file_path,
            img_idx,
        ]


def parse_csn_args():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3, help="The actor's learning rate.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4, help="The value of N in N-step A2C.")
    parser.add_argument("--class-weights", dest="class_weights", type=list, default=None, help="")

    parser.add_argument("--model-type", dest="model_type", type=str, default=None, help="")  # TODO: no used for now

    return parser.parse_args()


def main_train():
    model = CorrectSegNet(
        lr=config.lr,
        batch_size=config.batch_size,
        class_weights=config.class_weights,
        model_type=config.model_type,
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=config.num_epochs,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )
    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    config = parse_csn_args()
    main_train()
