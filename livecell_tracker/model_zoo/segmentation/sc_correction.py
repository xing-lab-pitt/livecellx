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

TEST_LOADER_IN_VAL_LOADER_LIST_IDX = 1


class CorrectSegNet(LightningModule):
    def __init__(
        self,
        lr=1e-3,
        batch_size=5,
        class_weights=[1, 1, 1],
        model_type=None,
        num_workers=16,
        train_input_paths=None,
        train_transforms=None,
        seed=99,
        train_dataset: CorrectSegNetDataset = None,
        val_dataset: CorrectSegNetDataset = None,
        test_dataset: CorrectSegNetDataset = None,
        kernel_size=(1, 1),
        num_classes=3,
        loss_type="CE",
        # the following args handled by dataset class
        input_type="raw_aug_seg",  # WARNING: do not change: be consistent with dataset class
        apply_gt_seg_edt=False,
        exclude_raw_input_bg=False,
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
        input_type : str, optional
            WARNING: do not change this default value
            input_type arg here should be consistent with what is used in dataset class
        """

        super().__init__()
        self.save_hyperparameters()
        self.generator = torch.Generator().manual_seed(seed)
        self.class_weights = class_weights
        self.model_type = model_type
        # self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DeepLabV3_ResNet50_Weights.DEFAULT")
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=kernel_size, stride=(1, 1))

        self.loss_type = loss_type
        self.loss_func = None
        if self.loss_type == "CE":
            print(">>> Using CE loss, weights:", self.class_weights)
            self.loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights))
            self.threshold = 0
        elif self.loss_type == "MSE":
            print(">>> Using MSE loss")
            self.loss_func = torch.nn.MSELoss()
            self.threshold = 1  # edt dist
        elif self.loss_type == "BCE":
            print(">>> Using BCE loss with logits loss")
            self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weights))
            self.threshold = 0
        else:
            raise NotImplementedError("Loss:%s not implemented", loss_type)

        print(">>> Based on loss type, training output threshold: ", self.threshold)
        self.learning_rate = lr
        self.batch_size = batch_size

        self.dims = (1, 412, 412)
        self.num_workers = num_workers
        self.train_input_paths = train_input_paths
        self.train_transforms = train_transforms
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # the following attributes not used; handled by dataset class
        self.apply_gt_seg_edt = apply_gt_seg_edt
        self.input_type = input_type
        self.exclude_raw_input_bg = exclude_raw_input_bg

    def forward(self, x: torch.Tensor):
        # print("[in forward] x shape: ", x.shape)
        x = self.model(x)
        x = x["out"]
        # return nn.functional.softmax(x, dim=1)
        return x

    def compute_loss(self, output: torch.tensor, target: torch.tensor):
        """Compute loss fuction

        Parameters
        ----------
        output : torch.tensor
            prediction output with shape batch_size x num_classes x height x width
        target : torch.tensor
            _description_

        Returns
        -------
        _type_
            _description_
        """
        assert len(output.shape) == 4
        if self.loss_type == "CE":
            return self.loss_func(output, target)
        elif self.loss_type == "MSE":
            total_loss = 0
            num_classes = output.shape[1]
            for cat_dim in range(0, num_classes):
                temp_target = target[:, cat_dim, ...]
                temp_output = output[:, cat_dim, ...]
                total_loss += self.loss_func(temp_output, temp_target) * self.class_weights[cat_dim]
            return total_loss
        elif self.loss_type == "BCE":
            output = output.permute(0, 2, 3, 1)
            target = target.permute(0, 2, 3, 1)
            return self.loss_func(output, target)

    def training_step(self, batch, batch_idx):
        # print("[train_step] x shape: ", batch["input"].shape)
        # print("[train_step] y shape: ", batch["gt_mask"].shape)
        x, y = batch["input"], batch["gt_mask"]
        output = self(x)
        loss = self.compute_loss(output, y)
        predicted_labels = torch.argmax(output, dim=1)
        self.log("train_loss", loss, batch_size=self.batch_size)
        

        # monitor more stats during training
        # compute on subdirs
        if self.global_step % 1000 != 0:
            return loss
        subdir_set = self.train_dataset.subdir_set
        batch_subdirs = np.array([self.train_dataset.get_subdir(idx.item()) for idx in batch["idx"]])
        bin_output = self.compute_bin_output(output)
        acc = self.train_accuracy(bin_output.long(), y.long())
        self.log("train_acc", acc, prog_bar=True)


        for subdir in subdir_set:
            if not (subdir in batch_subdirs):
                continue
            subdir_indexer = batch_subdirs == subdir
            batched_loss = self.compute_loss(output[subdir_indexer], y[subdir_indexer])
            # subdir_loss_map[subdir] = loss[list(subdir_indexer)].mean()
            self.log(f"train_loss_{subdir}", batched_loss, prog_bar=True)
            batched_acc = self.val_accuracy(bin_output[subdir_indexer].long(), y[subdir_indexer].long())
            self.log(f"train_acc_{subdir}", batched_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # print("[validation_step] x shape: ", batch["input"].shape)
        # print("[validation_step] y shape: ", batch["gt_mask"].shape)
        cur_loader = self.val_loaders[dataloader_idx]
        if dataloader_idx == TEST_LOADER_IN_VAL_LOADER_LIST_IDX:
            self.test_step(batch, batch_idx)
            return
        x, y = batch["input"], batch["gt_mask"]
        output = self(x)
        loss = self.compute_loss(output, y)
        # print("[val acc update] output shape: ", output.shape)
        # print("[val acc update] y shape: ", y.shape)

        # TODO: predicted_labels for pytorch ignite version accuracy
        # predicted_labels = torch.argmax(output, dim=1)
        # print("[val acc update] predicted_labels shape: ", predicted_labels.shape)
        # self.val_accuracy.update(predicted_labels.long(), y.long())
        bin_output = self.compute_bin_output(output)
        acc = self.val_accuracy(bin_output.long(), y.long())
        self.log("val_acc", acc, prog_bar=True, batch_size=self.batch_size)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        from livecell_tracker.model_zoo.segmentation.eval_csn import compute_metrics

        x, y = batch["input"], batch["gt_mask"]
        output = self(x)
        loss = self.compute_loss(output, y)
        self.log("test_loss", loss, prog_bar=True)
        bin_output = self.compute_bin_output(output)
        # subset test loss and acc according to self.subdirs
        subdir_set = self.test_dataset.subdir_set
        batch_subdirs = np.array([self.test_dataset.get_subdir(idx.item()) for idx in batch["idx"]])
        for subdir in subdir_set:
            if not (subdir in batch_subdirs):
                continue
            subdir_indexer = batch_subdirs == subdir
            batched_loss = self.compute_loss(output[subdir_indexer], y[subdir_indexer])
            # subdir_loss_map[subdir] = loss[list(subdir_indexer)].mean()
            self.log(f"test_loss_{subdir}", batched_loss, prog_bar=True, add_dataloader_idx=False)
            batched_acc = self.val_accuracy(bin_output[subdir_indexer].long(), y[subdir_indexer].long())
            self.log(f"test_acc_{subdir}", batched_acc, prog_bar=True, add_dataloader_idx=False)

            # Assemble batch for compute_metrics
            # get batch based on subdir_indexer
            subdir_batch = {}
            for key in batch.keys():
                subdir_batch[key] = batch[key][subdir_indexer]
            num_samples = len(subdir_batch["input"])
            subdir_samples = []
            for i in range(num_samples):
                tmp_dict = {}
                for key in subdir_batch.keys():
                    tmp_dict[key] = subdir_batch[key][i].cpu()
                subdir_samples.append(tmp_dict)

            metrics_dict = compute_metrics(
                subdir_samples,
                self,
                out_threshold=self.threshold,
                gt_label_masks=subdir_batch["gt_label_mask"].cpu().numpy(),
            )
            log_metrics = ["out_matched_num_gt_iou_0.5_percent", "out_matched_num_gt_iou_0.8_percent"]
            for metric in log_metrics:
                self.log(
                    f"test_{metric}_{subdir}", np.mean(metrics_dict[metric]), prog_bar=True, add_dataloader_idx=False
                )

    def compute_bin_output(self, output):
        output = output.clone()  # avoid inplace operation during training
        if self.loss_type == "CE" or self.loss_type == "BCE":
            # take sigmoid
            sigmoid_output = torch.sigmoid(output)
            output[output > self.threshold] = 1
            output[output <= self.threshold] = 0
            return sigmoid_output
        elif self.loss_type == "MSE":
            output[output > self.threshold] = 1
            output[output <= self.threshold] = 0
            return output

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
        self.val_loaders = [
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                generator=self.generator,
            ),
            self.test_dataloader(),
        ]
        return self.val_loaders

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

    def output_to_logits(self, out_tensor):
        """convert output tensors to logits"""
        out_tensor = torch.sigmoid(out_tensor)
        return out_tensor


def parse_csn_args():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3, help="The actor's learning rate.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4, help="The value of N in N-step A2C.")

    parser.add_argument("--model-type", dest="model_type", type=str, default=None, help="")  # TODO: no used for now

    return parser.parse_args()


if __name__ == "__main__":
    pass
