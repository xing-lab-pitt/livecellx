from pathlib import Path
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import models
import pytorch_lightning as pl
from dataset import CustomDataset, DataModule
from model import ViTModel


import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--model_dir", type=str, required=False)
parser.add_argument("--data_dir", type=str, default="../notebook_results/mmaction_train_data_v14-inclusive-imgs")
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="ViT_workdirs/eval_results/")
parser.add_argument("--frame-type", type=str, default="all")

args = parser.parse_args()

# Load the best model checkpoint
model_checkpoint_path = args.ckpt
model = ViTModel.load_from_checkpoint(model_checkpoint_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Load your test DataFrame
test_df = pd.read_csv(Path(args.data_dir) / "test.csv")

# Filter based on frame type
if args.frame_type != "all":
    print("filtering based on frame_type:", args.frame_type)
    test_df = test_df[test_df["frame_type"] == args.frame_type]

# Assuming you have the same Transform from your training script
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.5, 1.5)),
    ]
)

# Create a DataLoader for the test set
test_dataset = CustomDataset(test_df, transform=transform, data_dir=Path(args.data_dir))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=32)

# Evaluate the model
import tqdm

predictions = []
with torch.no_grad():
    for inputs, _ in tqdm.tqdm(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())

# Add predictions to the test DataFrame
test_df["predictions"] = predictions


# Optionally, if you want to save this DataFrame
Path(args.out_dir).mkdir(exist_ok=True, parents=True)
test_df.to_csv(Path(args.out_dir) / "test_predictions.csv", index=False)

# Display the DataFrame
print(test_df.head())
