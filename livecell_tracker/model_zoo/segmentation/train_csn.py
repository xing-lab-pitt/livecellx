import torch

torch.manual_seed(237)
import argparse
from pathlib import Path
import pandas as pd
from torchvision import transforms
import torch
import torch.utils.data
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from livecell_tracker.model_zoo.segmentation.sc_correction import CorrectSegNet
from livecell_tracker.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Correct Segmentation Net Training")
    parser.add_argument("--train_dir", dest="train_dir", type=str, required=True)
    parser.add_argument("--kernel_size", dest="kernel_size", type=int, default=1)
    parser.add_argument("--model", dest="model_file", type=str)
    parser.add_argument("--lr", dest="lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=2)
    parser.add_argument("--translation", dest="translation", type=float, default=0.5)
    parser.add_argument("--degrees", dest="degrees", type=int, default=180)
    parser.add_argument("--aug_scale", dest="aug_scale", type=str, default="0.5,1.5")
    parser.add_argument("--split_seed", dest="split_seed", type=int, default=237)

    args = parser.parse_args()
    args.aug_scale = [float(x) for x in args.aug_scale.split(",")]
    return args


def main_train():
    args = parse_args()
    # train_dir = Path("./notebook_results/a549_ccp_vim/train_data_v1")
    train_dir = Path(args.train_dir)
    train_csv = train_dir / "train_data.csv"
    kernel_size = args.kernel_size
    train_df = pd.read_csv(train_csv)
    print("pd df shape:", train_df.shape)
    print("df samples:", train_df[:2])

    # augmentation params
    translation_range = (args.translation, args.translation)
    degrees = args.degrees

    raw_img_paths = list(train_df["raw"])
    scaled_seg_mask_paths = list(train_df["seg"])
    gt_mask_paths = list(train_df["gt"])
    raw_seg_paths = list(train_df["raw_seg"])
    scales = list(train_df["scale"])
    aug_diff_img_paths = list(train_df["aug_diff_mask"])
    raw_transformed_img_paths = list(train_df["raw_transformed_img"])

    train_transforms = transforms.Compose(
        [
            # transforms.Resize((412, 412)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=degrees, translate=translation_range, scale=args.aug_scale),
            transforms.RandomCrop((412, 412), pad_if_needed=True),
        ]
    )

    dataset = CorrectSegNetDataset(
        raw_img_paths,
        scaled_seg_mask_paths,
        gt_mask_paths,
        raw_seg_paths=raw_seg_paths,
        scales=scales,
        transform=train_transforms,
        raw_transformed_img_paths=raw_transformed_img_paths,
        aug_diff_img_paths=aug_diff_img_paths,
    )

    train_sample_num = int(len(dataset) * 0.8)
    val_sample_num = len(dataset) - train_sample_num
    split_generator = torch.Generator().manual_seed(args.split_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_sample_num, val_sample_num], generator=split_generator
    )

    model = CorrectSegNet(
        train_input_paths=train_input_tuples,
        num_workers=1,
        batch_size=args.batch_size,
        train_transforms=train_transforms,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=val_dataset,
        kernel_size=kernel_size,
    )
    trainer = Trainer(gpus=1, max_epochs=500)
    trainer.fit(model)


if __name__ == "__main__":
    main_train()
