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
import numpy as np
from livecell_tracker.preprocess.utils import normalize_img_to_uint8

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--raw_dir", type=str, required=True, help="Path to the directory containing the raw images")
parser.add_argument("--mask_dir", type=str, required=True, help="Path to the directory containing the EDT masks")
parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint file")
args = parser.parse_args()

print("Command line arguments:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

# Define the transforms for the raw images and masks
raw_transforms = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)

mask_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


class RawAndEdtMaskDataset(Dataset):
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

        # raw_image = Image.open(raw_path).convert("L")
        # mask_image = Image.open(mask_path).convert("L")
        raw_image = np.array(Image.open(raw_path))
        edt_mask_image = np.array(Image.open(mask_path))
        raw_image = normalize_img_to_uint8(raw_image)
        edt_mask_image = np.minimum(edt_mask_image, 255)

        # transform to tensor
        raw_image = Image.fromarray(raw_image)
        edt_mask_image = Image.fromarray(edt_mask_image)
        if self.transform:
            raw_image = self.transform(raw_image)
            edt_mask_image = self.transform(edt_mask_image)

        return raw_image, edt_mask_image


# Define the datasets and dataloaders
raw_dataset = RawAndEdtMaskDataset(
    args.raw_dir,
    args.mask_dir,
    transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
)

train_dataset, val_dataset = torch.utils.data.random_split(
    raw_dataset, [int(0.8 * len(raw_dataset)), len(raw_dataset) - int(0.8 * len(raw_dataset))]
)
sample_raw_img, sample_mask_img = raw_dataset[0]
print(f"Raw image shape: {sample_raw_img.shape}")
print(f"Mask image shape: {sample_mask_img.shape}")

# save to disk for debugging
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)
ax[0].imshow(sample_raw_img[0])
ax[1].imshow(sample_mask_img[0])
plt.savefig("./sample_raw_mask.png")

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Define the PyTorch Lightning module and trainer
model = RegSegModel()

checkpoint_callback = ModelCheckpoint(monitor="val_iou", mode="max", save_top_k=5)

trainer = Trainer(
    gpus=1,
    max_epochs=10000,
    val_check_interval=100,
    checkpoint_callback=checkpoint_callback,
    resume_from_checkpoint=args.ckpt_path,
)

# Train the model
trainer.fit(model, train_dataloader, val_dataloader)
