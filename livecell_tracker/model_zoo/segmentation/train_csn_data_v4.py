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
from pytorch_lightning.loggers import TensorBoardLogger
from livecell_tracker.model_zoo.segmentation.sc_correction import CorrectSegNet
from livecell_tracker.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Correct Segmentation Net Training")
    parser.add_argument("--train_dir", dest="train_dir", type=str, required=True)
    parser.add_argument(
        "--model_version",
        dest="model_version",
        type=str,
        default=None,
        help="The model version. Used in tensorboard to save your model file and train/val/test results.",
    )
    parser.add_argument("--kernel_size", dest="kernel_size", type=int, default=1)
    parser.add_argument("--model", dest="model_file", type=str)
    parser.add_argument("--model_ckpt", dest="model_ckpt", type=str, default=None)
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=2)
    parser.add_argument("--translation", dest="translation", type=float, default=0.5)
    parser.add_argument("--degrees", dest="degrees", type=int, default=180)
    parser.add_argument("--aug_scale", dest="aug_scale", type=str, default="0.5,1.5")
    parser.add_argument("--split_seed", dest="split_seed", type=int, default=237)
    parser.add_argument("--epochs", dest="epochs", type=int, default=1000)

    parser.add_argument(
        "--input_type",
        dest="input_type",
        type=str,
        default="raw_aug_seg",
        choices=["raw_aug_seg", "raw_aug_duplicate", "raw_duplicate", "edt_v0"],
    )
    parser.add_argument("--apply_gt_seg_edt", dest="apply_gt_seg_edt", default=False, action="store_true")
    parser.add_argument("--class-weights", dest="class_weights", type=str, default="1,1,1")
    parser.add_argument("--loss", dest="loss", type=str, default="CE", choices=["CE", "MSE", "BCE"])
    parser.add_argument("--exclude_raw_input_bg", dest="exclude_raw_input_bg", default=False, action="store_true")

    parser.add_argument("--debug", dest="debug", default=False, action="store_true")
    args = parser.parse_args()

    # convert string to list
    args.aug_scale = [float(x) for x in args.aug_scale.split(",")]
    args.class_weights = [float(x) for x in args.class_weights.split(",")]
    return args


def main_train():
    args = parse_args()
    print("[Args] ", args)
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
    gt_label_mask_paths = list(train_df["gt_label_mask"])

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
        gt_label_mask_paths=gt_label_mask_paths,
        raw_seg_paths=raw_seg_paths,
        scales=scales,
        transform=train_transforms,
        raw_transformed_img_paths=raw_transformed_img_paths,
        aug_diff_img_paths=aug_diff_img_paths,
        input_type=args.input_type,
        apply_gt_seg_edt=args.apply_gt_seg_edt,
        exclude_raw_input_bg=args.exclude_raw_input_bg,
    )

    train_sample_num = int(len(dataset) * 0.8)
    val_sample_num = len(dataset) - train_sample_num
    split_generator = torch.Generator().manual_seed(args.split_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_sample_num, val_sample_num], generator=split_generator
    )

    model = CorrectSegNet(
        # train_input_paths=train_input_tuples,
        lr=args.lr,
        num_workers=1,
        batch_size=args.batch_size,
        train_transforms=train_transforms,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=val_dataset,
        kernel_size=kernel_size,
        loss_type=args.loss,
        class_weights=args.class_weights,
        # only for record keeping purposes; handled by the dataset
        input_type=args.input_type,
        apply_gt_seg_edt=args.apply_gt_seg_edt,
        exclude_raw_input_bg=args.exclude_raw_input_bg,
    )
    logger = TensorBoardLogger(save_dir=".", name="lightning_logs", version=args.model_version)
    if args.debug:
        logger = TensorBoardLogger(save_dir=".", name="test_logs", version=args.model_version)

    print("logger save dir:", logger.save_dir)
    print("logger subdir:", logger.sub_dir)
    print("logger version:", logger.version)
    trainer = Trainer(gpus=1, max_epochs=args.epochs, resume_from_checkpoint=args.model_ckpt, logger=logger)
    trainer.fit(model)


if __name__ == "__main__":
    main_train()
