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
from livecellx.core.datasets import LiveCellImageDataset
from livecellx.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset

TEST_LOADER_IN_VAL_LOADER_LIST_IDX = 1

LOG_PROGRESS_BAR = False

import torch
import torch.nn.functional as F


def weighted_mse_loss(predict, target, weights=None):
    """
    Compute the weighted MSE loss with an optional weight map for the first channel.

    Parameters:
    - input: Tensor of predicted values (batch_size, channels, height, width).
    - target: Tensor of target values with the same shape as input.
    - weights: Optional. Tensor of weights for the first channel (batch_size, 1, height, width).
               If None, no weights are applied and standard MSE loss is calculated.

    Returns:
    - loss: Scalar tensor representing the weighted MSE loss.
    """
    if weights is not None:
        # Calculate squared differences
        squared_diff = (predict - target) ** 2

        # Apply weights
        weighted_squared_diff = squared_diff * weights

        # Calculate mean of the weighted squared differences
        loss = weighted_squared_diff.mean()
    else:
        # If no weights are provided, calculate standard MSE loss
        loss = F.mse_loss(predict, target, reduction="mean")

    return loss


class CorrectSegNetAux(LightningModule):
    def __init__(
        self,
        lr=1e-3,
        batch_size=5,
        class_weights=[1, 1, 1],
        model_type=None,
        num_workers=32,
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
        normalize_uint8=False,
        log_progress_bar=LOG_PROGRESS_BAR,
        use_aux=True,
        aux_loss_weight=0.5,
        lr_scheduler_type=None,
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

        # Auxiliary classifier
        self.aux_classifier = None
        self.aux_loss_weight = aux_loss_weight
        self.use_aux = use_aux
        if use_aux:
            self.aux_classifier = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(256, 256, kernel_size=1),  # 3 classes
                nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
                nn.Flatten(),  # flatten the tensor
                nn.Linear(256, 4),  # fully connected layer for 4 classes
            )
        self.loss_type = loss_type
        self.loss_func = None
        if self.loss_type == "CE":
            print(">>> Using CE loss, weights:", self.class_weights)
            self.loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights))
            self.threshold = 0.5
        elif self.loss_type == "MSE":
            print(">>> Using MSE loss")
            self.loss_func = torch.nn.MSELoss()
            self.threshold = 1  # edt dist
        elif self.loss_type == "BCE":
            print(">>> Using BCE loss with logits loss")
            self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weights))
            self.threshold = 0.5
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
        self.normalize_uint8 = normalize_uint8

        # the following attributes not used; handled by dataset class
        self.apply_gt_seg_edt = apply_gt_seg_edt
        self.input_type = input_type
        self.exclude_raw_input_bg = exclude_raw_input_bg

        self.log_progress_bar = log_progress_bar
        self.lr_scheduler_type = lr_scheduler_type

    def forward(self, x: torch.Tensor):
        # print("[in forward] x shape: ", x.shape)
        input_shape = x.shape[-2:]
        ori_x = x.clone()
        x = self.model(x)
        x = x["out"]
        # return nn.functional.softmax(x, dim=1)
        if self.use_aux:
            features = self.model.backbone(ori_x)
            aux_result = self.aux_classifier(features["aux"])
            # aux_result = nn.functional.interpolate(aux_result, size=input_shape, mode='bilinear', align_corners=False)
            return x, aux_result
        else:
            return x

    def compute_loss(
        self, output: torch.tensor, target: torch.tensor, aux_out=None, aux_target=None, gt_pixel_weight=None
    ):
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

        aux_loss = 0
        seg_output = output
        if aux_out is not None and aux_target is not None:
            # Calculate crossEntropyLoss
            aux_loss = F.cross_entropy(aux_out, aux_target)

        assert (
            len(seg_output.shape) == 4
        ), "seg_output shape should be batch_size x num_classes x height x width, got %s" % str(seg_output.shape)

        if self.loss_type == "CE":
            seg_loss = self.loss_func(seg_output, target)
        elif self.loss_type == "MSE":
            total_loss = 0
            num_classes = seg_output.shape[1]
            for cat_dim in range(0, num_classes):
                temp_target = target[:, cat_dim, ...]
                temp_output = seg_output[:, cat_dim, ...]
                total_loss += (
                    weighted_mse_loss(temp_output, temp_target, weights=gt_pixel_weight) * self.class_weights[cat_dim]
                )
            seg_loss = total_loss
        elif self.loss_type == "BCE":
            # # Debugging
            # print("*" * 40)
            # print("Dimensions:")
            # print("seg_output shape: ", seg_output.shape)
            # print("target shape: ", target.shape)
            # print("*" * 40)
            # if gt_pixel_weight is not None:
            #     print("gt_pixel_weight shape: ", gt_pixel_weight.shape)
            if gt_pixel_weight is not None:
                # Repeat to match 3 channels of gt (seg and two OU masks): gt_pixel_weight shape: 2, 412, 412 -> 2, 3, 412, 412
                gt_pixel_weight_repeated = gt_pixel_weight.unsqueeze(1).repeat(1, 3, 1, 1)
                # assert len(gt_pixel_weight_repeated.shape) == 4
                gt_pixel_weight_permuted = gt_pixel_weight_repeated.permute(0, 2, 3, 1)
            else:
                gt_pixel_weight_permuted = None
            seg_output = seg_output.permute(0, 2, 3, 1)
            target = target.permute(0, 2, 3, 1)
            self.loss_func = torch.nn.BCEWithLogitsLoss(
                weight=gt_pixel_weight_permuted, pos_weight=torch.tensor(self.class_weights).cuda()
            )

            seg_loss = self.loss_func(seg_output, target)
        else:
            raise NotImplementedError("Loss:%s not implemented", self.loss_type)

        return seg_loss, aux_loss

    def training_step(self, batch, batch_idx):
        # print("[train_step] x shape: ", batch["input"].shape)
        # print("[train_step] y shape: ", batch["gt_mask"].shape)
        x, y = batch["input"], batch["gt_mask"]
        aux_target = batch["ou_aux"]
        gt_pixel_weight = batch["gt_pixel_weight"]
        output, aux_out = self(x)
        seg_loss, aux_loss = self.compute_loss(
            output, y, aux_out=aux_out, aux_target=aux_target, gt_pixel_weight=gt_pixel_weight
        )
        loss = seg_loss + self.aux_loss_weight * aux_loss
        predicted_labels = torch.argmax(output, dim=1)
        self.log(
            "train_loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=self.log_progress_bar
        )
        self.log(
            "train_seg_loss",
            seg_loss,
            batch_size=self.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=self.log_progress_bar,
        )
        self.log(
            "train_aux_loss",
            aux_loss,
            batch_size=self.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=self.log_progress_bar,
        )

        aux_predicted_labels = torch.argmax(aux_out, dim=1)
        aux_true_labels = torch.argmax(aux_target, dim=1)

        # calculate accuracy manually
        correct_predictions = (aux_predicted_labels == aux_true_labels).float()
        aux_acc = correct_predictions.sum() / len(correct_predictions)

        # log the auxiliary accuracy
        self.log(
            "train_aux_acc",
            aux_acc,
            prog_bar=self.log_progress_bar,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        # monitor more stats during training
        # compute on subdirs
        if self.global_step % 1000 != 0:
            return loss
        subdir_set = self.train_dataset.subdir_set
        batch_subdirs = np.array([self.train_dataset.get_subdir(idx.item()) for idx in batch["idx"]])
        bin_output = self.compute_bin_output(output)
        acc = self.train_accuracy(bin_output.long(), y.long())
        self.log(
            "train_acc", acc, prog_bar=self.log_progress_bar, on_step=True, on_epoch=True, batch_size=self.batch_size
        )

        for subdir in subdir_set:
            if not (subdir in batch_subdirs):
                continue
            subdir_indexer = batch_subdirs == subdir
            seg_batched_loss, aux_batched_loss = self.compute_loss(
                output[subdir_indexer],
                y[subdir_indexer],
                aux_out=aux_out[subdir_indexer],
                aux_target=aux_target[subdir_indexer],
            )
            # subdir_loss_map[subdir] = loss[list(subdir_indexer)].mean()
            self.log(f"train_seg_loss_{subdir}", seg_batched_loss, prog_bar=self.log_progress_bar)
            self.log(f"train_aux_loss_{subdir}", aux_batched_loss, prog_bar=self.log_progress_bar)
            batched_acc = self.val_accuracy(bin_output[subdir_indexer].long(), y[subdir_indexer].long())
            self.log(f"train_acc_{subdir}", batched_acc, prog_bar=self.log_progress_bar)
        return loss

    def training_epoch_end(self, outputs):
        # calculate the epoch-level loss using the outputs list
        epoch_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # log the epoch loss
        self.log("train_loss_epoch", epoch_loss, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # print("[validation_step] x shape: ", batch["input"].shape)
        # print("[validation_step] y shape: ", batch["gt_mask"].shape)
        cur_loader = self.val_loaders[dataloader_idx]
        if dataloader_idx == TEST_LOADER_IN_VAL_LOADER_LIST_IDX:
            self.test_step(batch, batch_idx)
            return
        x, y = batch["input"], batch["gt_mask"]
        aux_target = batch["ou_aux"]
        output, aux_out = self(x)
        seg_loss, aux_loss = self.compute_loss(output, y, aux_out=aux_out, aux_target=aux_target)
        loss = seg_loss + self.aux_loss_weight * aux_loss
        # print("[val acc update] output shape: ", output.shape)
        # print("[val acc update] y shape: ", y.shape)

        # TODO: predicted_labels for pytorch ignite version accuracy
        # predicted_labels = torch.argmax(output, dim=1)
        # print("[val acc update] predicted_labels shape: ", predicted_labels.shape)
        # self.val_accuracy.update(predicted_labels.long(), y.long())
        bin_output = self.compute_bin_output(output)
        acc = self.val_accuracy(bin_output.long(), y.long())
        self.log("val_acc", acc, prog_bar=self.log_progress_bar, batch_size=self.batch_size, add_dataloader_idx=False)
        self.log("val_loss", loss, prog_bar=self.log_progress_bar, add_dataloader_idx=False)
        self.log("val_seg_loss", seg_loss, prog_bar=self.log_progress_bar, add_dataloader_idx=False)
        self.log("val_aux_loss", aux_loss, prog_bar=self.log_progress_bar, add_dataloader_idx=False)

        aux_predicted_labels = torch.argmax(aux_out, dim=1)
        aux_true_labels = torch.argmax(aux_target, dim=1)

        # calculate accuracy manually
        correct_predictions = (aux_predicted_labels == aux_true_labels).float()
        aux_acc = correct_predictions.sum() / len(correct_predictions)

        # log the auxiliary accuracy
        self.log(
            "val_aux_acc",
            aux_acc,
            prog_bar=self.log_progress_bar,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def test_step(self, batch, batch_idx):
        from livecellx.model_zoo.segmentation.eval_csn import compute_metrics

        x, y = batch["input"], batch["gt_mask"]
        aux_target = batch["ou_aux"]
        output, aux_out = self(x)
        seg_loss, aux_loss = self.compute_loss(output, y, aux_out=aux_out, aux_target=aux_target)
        loss = seg_loss + self.aux_loss_weight * aux_loss

        self.log("test_loss", loss, prog_bar=self.log_progress_bar, add_dataloader_idx=False)
        bin_output = self.compute_bin_output(output)
        # subset test loss and acc according to self.subdirs
        subdir_set = self.test_dataset.subdir_set
        batch_subdirs = np.array([self.test_dataset.get_subdir(idx.item()) for idx in batch["idx"]])
        for subdir in subdir_set:
            if not (subdir in batch_subdirs):
                continue
            subdir_indexer = batch_subdirs == subdir
            seg_loss, aux_loss = self.compute_loss(output[subdir_indexer], y[subdir_indexer])
            batched_loss = seg_loss + self.aux_loss_weight * aux_loss
            # subdir_loss_map[subdir] = loss[list(subdir_indexer)].mean()
            self.log(f"test_loss_{subdir}", seg_loss, prog_bar=self.log_progress_bar, add_dataloader_idx=False)
            self.log(f"test_seg_loss_{subdir}", batched_loss, prog_bar=self.log_progress_bar, add_dataloader_idx=False)
            batched_acc = self.val_accuracy(bin_output[subdir_indexer].long(), y[subdir_indexer].long())
            self.log(f"test_acc_{subdir}", batched_acc, prog_bar=self.log_progress_bar, add_dataloader_idx=False)

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
            log_metrics = [
                "out_matched_num_gt_iou_0.5_percent",
                "out_matched_num_gt_iou_0.8_percent",
                "out_matched_num_gt_iou_0.95_percent",
            ]
            for metric in log_metrics:
                self.log(
                    f"test_{metric}_{subdir}",
                    np.mean(metrics_dict[metric]),
                    prog_bar=self.log_progress_bar,
                    add_dataloader_idx=False,
                )

    def compute_bin_output(self, output):
        output = output.clone()  # avoid inplace operation during training
        # output = [x.clone for x in output if x is not None]
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
        if self.lr_scheduler_type is None:
            pass
        elif self.lr_scheduler_type == "ReduceLROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}
        else:
            assert False, f"lr_scheduler_type:{self.lr_scheduler_type} not implemented"

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
        ]
        if self.test_dataset is not None:
            self.val_loaders.append(
                DataLoader(
                    self.test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    generator=self.generator,
                )
            )
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

        output, aux_out = self(x)
        return output

    def output_to_logits(self, out_tensor):
        """convert output tensors to logits for segmentation masks"""
        if self.use_aux:
            out_tensor = out_tensor[0]
        out_tensor = torch.sigmoid(out_tensor)
        return out_tensor

    def aux_output_to_logits(self, aux_out_tensor):
        """convert output tensors to logits for classifier results"""
        assert self.use_aux
        out_tensor = out_tensor[1]
        aux_out_tensor = torch.sigmoid(aux_out_tensor)
        return aux_out_tensor


def parse_csn_args():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3, help="The actor's learning rate.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4, help="The value of N in N-step A2C.")

    parser.add_argument("--model-type", dest="model_type", type=str, default=None, help="")  # TODO: no used for now

    return parser.parse_args()


if __name__ == "__main__":
    pass
