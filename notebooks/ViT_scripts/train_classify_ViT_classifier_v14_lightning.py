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

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--start_frame_idx", type=int, default=1)
parser.add_argument("--end_frame_idx", type=int, default=5)

parser.add_argument("--model_version", type=str, default="NoVersion")
parser.add_argument("--frame-type", type=str, default="all")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--ckpt", type=str, default=None)
args = parser.parse_args()


# Load your DataFrame
# Load your DataFrame
DATA_DIR = Path("../notebook_results/mmaction_train_data_v14-inclusive-imgs")
df = pd.read_csv(DATA_DIR / "train_and_test.csv")
out_dir = Path("ViT_workdirs")
out_dir.mkdir(exist_ok=True)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.5, 1.5)),
    ]
)


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_relative_path = self.dataframe.iloc[idx]["img_path"]
        img_path = DATA_DIR / img_relative_path
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


class ViTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.vit_b_16(pretrained=True)
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


# Split your dataset
train_df = df[df["split"] == "train"]
valid_df = df[df["split"] == "test"]

print("filtering based on start_frame_idx and end_frame_idx:", args.start_frame_idx)
print("before filtering, train_df.shape:", train_df.shape)
print("before filtering, valid_df.shape:", valid_df.shape)

# Filter based on start_frame_idx

train_df = train_df[train_df["frame_idx"] >= args.start_frame_idx]
valid_df = valid_df[valid_df["frame_idx"] >= args.start_frame_idx]

# Filter based on end_frame_idx
print("filtering based on end_frame_idx:", args.end_frame_idx)
train_df = train_df[train_df["frame_idx"] <= args.end_frame_idx]
valid_df = valid_df[valid_df["frame_idx"] <= args.end_frame_idx]

print("after filtering, train_df.shape:", train_df.shape)
print("after filtering, valid_df.shape:", valid_df.shape)


if args.frame_type != "all":
    print("filtering based on input type:", args.frame_type)
    print("before filtering, train_df.shape:", train_df.shape)
    print("before filtering, valid_df.shape:", valid_df.shape)

    train_df = train_df[train_df["frame_type"] == args.frame_type]
    valid_df = valid_df[valid_df["frame_type"] == args.frame_type]

    print("after filtering, train_df.shape:", train_df.shape)
    print("after filtering, valid_df.shape:", valid_df.shape)


# Debug: reduce the size of the dataset
if args.debug:
    train_df = train_df[:100]
    valid_df = valid_df[:10]

data_module = DataModule(train_df, valid_df, batch_size=args.batch_size)
model = ViTModel()

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    # filename="{epoch:02d}",
)
logger_name = "ViT_lightning_logs"
logger = TensorBoardLogger(save_dir=out_dir, version=args.model_version, name=logger_name)
print("logger save dir:", logger.save_dir)
print("logger subdir:", logger.sub_dir)
print("logger version:", logger.version)
# Trainer setup
trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback], default_root_dir=out_dir, gpus=1, logger=logger)

# Save all the arguments
args_dict = vars(args)
args_df = pd.DataFrame.from_dict(args_dict, orient="index")
args_csv_path = logger.save_dir / logger_name / logger.version / "args.csv"
args_csv_path.parent.mkdir(exist_ok=True, parents=True)
args_df.to_csv(args_csv_path)


# Train the model
trainer.fit(model, data_module)
