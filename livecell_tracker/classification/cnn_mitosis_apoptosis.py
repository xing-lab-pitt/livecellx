import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from livecell_tracker.preprocess.utils import normalize_img_to_uint8

DEFAULT_CLASS_NAMES = ["mitosis", "apoptosis", "normal"]


class MitApopImageDataset(Dataset):
    def __init__(
        self,
        dir_path=None,
        subdir_classes=DEFAULT_CLASS_NAMES,
        transform=None,
    ):
        self.dir_path = dir_path
        self.transform = transform

        self.image_files = []
        self.labels = []
        for label, folder_name in enumerate(subdir_classes):
            folder_path = os.path.join(dir_path, folder_name)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                self.image_files.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label = self.labels[idx]
        image = Image.open(image_path)
        image = normalize_img_to_uint8(np.array(image))
        # print("shape of image before hstack", image.shape)
        # image = np.stack([image, image, image])
        # # permute
        # print("shape of image before permute", image.shape)
        # image = np.transpose(image, (2, 0, 1))
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        image = np.array(image)
        image = torch.stack([torch.tensor(image), torch.tensor(image), torch.tensor(image)])
        return image.squeeze(), label


class MitApopImageClassifier(pl.LightningModule):
    def __init__(self, dir_path, n_classes=3, train_crop_size=224, val_dir_path=None, class_names=DEFAULT_CLASS_NAMES):
        super().__init__()
        self.dir_path = dir_path
        self.val_dir_path = val_dir_path
        self.train_crop_size = train_crop_size
        self.n_classes = n_classes
        self.class_names = class_names
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 14 * 14, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # check mean, std of x, y
        # print(x.mean(), x.std())
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y) / float(len(y))
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y) / float(len(y))
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

        # also record validation acc by class
        for i in range(3):
            idx = y == i
            if torch.sum(idx) > 0:
                acc = torch.sum(preds[idx] == y[idx]) / float(torch.sum(idx))
                self.log(f"val_acc_{self.class_names[i]}", acc, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.5, patience=1
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        #     "monitor": "train_loss",
        # }

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                # add zoom
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.5, 2)),
                transforms.RandomCrop(self.train_crop_size, pad_if_needed=True, padding_mode="constant"),
                # transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dataset = MitApopImageDataset(self.dir_path, transform=transform)
        return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                # transforms.RandomCrop(self.train_crop_size, pad_if_needed=True, padding_mode="constant"),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dataset = MitApopImageDataset(self.val_dir_path, transform=transform)
        return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
