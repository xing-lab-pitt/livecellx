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
        class_weights=[1, 1],
        model_type=None,
        num_workers=16,
        train_input_paths=None,
        train_transforms=None,
        seed=99,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        kernel_size=(1, 1),
        loss_func=nn.CrossEntropyLoss(),
        num_classes=3,
        # the following args handled by dataset class
        input_type=None,
        apply_gt_seg_edt=False,
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
        self.save_hyperparameters()
        self.generator = torch.Generator().manual_seed(seed)
        self.class_weights = torch.tensor(class_weights).cuda()
        self.model_type = model_type
        # self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DeepLabV3_ResNet50_Weights.DEFAULT")
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=kernel_size, stride=(1, 1))

        self.loss_func = loss_func
        self.learning_rate = lr
        self.batch_size = batch_size

        self.dims = (1, 412, 412)
        self.num_workers = num_workers
        self.train_input_paths = train_input_paths
        self.train_transforms = train_transforms

        self.save_hyperparameters()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def forward(self, x: torch.Tensor):
        # print("[in forward] x shape: ", x.shape)
        x = self.model(x)
        x = x["out"]
        # return nn.functional.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        # print("[train_step] x shape: ", batch["input"].shape)
        # print("[train_step] y shape: ", batch["gt_mask"].shape)
        x, y = batch["input"], batch["gt_mask"]
        output = self(x)
        loss = self.loss_func(output, y)

        predicted_labels = torch.argmax(output, dim=1)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        # print("[validation_step] x shape: ", batch["input"].shape)
        # print("[validation_step] y shape: ", batch["gt_mask"].shape)

        x, y = batch["input"], batch["gt_mask"]
        output = self(x)
        loss = self.loss_func(output, y)
        # print("[val acc update] output shape: ", output.shape)
        # print("[val acc update] y shape: ", y.shape)

        # TODO: predicted_labels for pytorch ignite version accuracy
        # predicted_labels = torch.argmax(output, dim=1)
        # print("[val acc update] predicted_labels shape: ", predicted_labels.shape)
        # self.val_accuracy.update(predicted_labels.long(), y.long())

        self.val_accuracy.update(output, y.long())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        x, y = batch["input"], batch["gt_mask"]

        output = self(x)
        loss = self.loss_func(output, y)
        self.test_accuracy.update(output, y.long())

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage=None):
        ################### datasets settings ###################
        # img_paths, mask_paths, gt_paths = list(zip(*self.train_input_paths))
        pass

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

        output = self(x)
        return output


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
