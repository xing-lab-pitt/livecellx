import argparse
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from livecell_tracker.model_zoo.segmentation.wwk_reg_seg_model import RegSegModel
from PIL import Image

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--raw_dir", type=str, required=True, help="Path to the directory containing the raw images")
parser.add_argument("--mask_dir", type=str, required=True, help="Path to the directory containing the EDT masks")
args = parser.parse_args()

# Define the transforms for the raw images and masks
raw_transforms = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)

mask_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
# Define the transforms for the raw images and masks
raw_transforms = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)

mask_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


class RawMaskDataset(Dataset):
    def __init__(self, raw_dir, mask_dir, transform=None):
        self.raw_dir = raw_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.raw_filenames = sorted(os.listdir(raw_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.raw_filenames)

    def __getitem__(self, idx):
        raw_path = os.path.join(self.raw_dir, self.raw_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        raw_image = Image.open(raw_path).convert("L")
        mask_image = Image.open(mask_path).convert("L")

        if self.transform:
            raw_image = self.transform(raw_image)
            mask_image = self.transform(mask_image)

        return raw_image, mask_image


# Define the datasets and dataloaders
raw_dataset = RawMaskDataset(
    args.raw_dir,
    args.mask_dir,
    transform=transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    ),
)

dataloader = DataLoader(raw_dataset, batch_size=4, shuffle=True)

# Define the PyTorch Lightning module and trainer
model = RegSegModel()
trainer = Trainer(gpus=1, max_epochs=10, checkpoint_callback=ModelCheckpoint(monitor="val_loss"))

# Train the model
trainer.fit(model, dataloader)
