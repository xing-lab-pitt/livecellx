from pathlib import Path
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import models
import pytorch_lightning as pl
from dataset import CustomDataset, DataModule
from model import LcaImageClassificationModel


import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--model_dir", type=str, required=False)
parser.add_argument("--data_dir", type=str, default="../notebook_results/mmaction_train_data_v14-inclusive-imgs")
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="ViT_workdirs/eval_results/")
parser.add_argument("--frame-type", type=str, default="all")
parser.add_argument("--affine-aug", action="store_true")
parser.add_argument("--save_dir_suffix", type=str, default="")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--model", type=str, default="vit_b_16", help="vit_b_16 or resnet50", choices=["vit_b_16", "resnet50"]
)

args = parser.parse_args()
print("=" * 40)
print("args:")
for k, v in vars(args).items():
    print(f"{k}: {v}")
print("=" * 40)

# Load the best model checkpoint
model_checkpoint_path = args.ckpt

# If model_checkpoint_path is None, then load the best checkpoint automatically
# The checkouts are in {args.data_dir}/{args.model_dir}/checkpoints/* with format epoch={epoch}-step={step}.ckpt load the best checkpoint automatically

import glob

if model_checkpoint_path is None:
    model_checkpoint_path = sorted(glob.glob(f"{args.model_dir}/checkpoints/*.ckpt"))[-1]
    print("loading checkpoint automatically:", model_checkpoint_path)


model = LcaImageClassificationModel.load_from_checkpoint(model_checkpoint_path, model=args.model)
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
augs = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

if args.affine_aug:
    augs += [transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.5, 1.5))]
transform = transforms.Compose(augs)

# Create a DataLoader for the test set
test_dataset = CustomDataset(test_df, transform=transform, data_dir=Path(args.data_dir))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32)

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

model_name = Path(args.model_dir).name

if args.affine_aug:
    model_name += "-affine-aug"

model_out_dir = Path(args.out_dir) / (str(model_name) + args.save_dir_suffix)
model_out_dir.mkdir(exist_ok=True, parents=True)
test_df.to_csv(model_out_dir / "test_predictions.csv", index=False)

# Display the DataFrame
print(test_df.head())


# Save all the wrong predictions separately
wrong_predictions_df = test_df[test_df["label_index"] != test_df["predictions"]]
wrong_predictions_df.to_csv(model_out_dir / "wrong_predictions.csv", index=False)

# Calculate accuracy, precision, recall, and F1 score and store them in a df
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics_df = pd.DataFrame(
    {
        "accuracy": [accuracy_score(test_df["label_index"], test_df["predictions"])],
        "precision": [precision_score(test_df["label_index"], test_df["predictions"], average="macro")],
        "recall": [recall_score(test_df["label_index"], test_df["predictions"], average="macro")],
        "f1": [f1_score(test_df["label_index"], test_df["predictions"], average="macro")],
    }
)
metrics_df.to_csv(model_out_dir / "metrics.csv", index=False)
