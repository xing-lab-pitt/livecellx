import torch

torch.manual_seed(237)
import argparse
from pathlib import Path
import pandas as pd

import torch
import torch.utils.data
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import livecellx
from livecellx.model_zoo.segmentation.sc_correction import CorrectSegNet
from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux
from livecellx.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset
import livecellx.model_zoo.segmentation.csn_configs as csn_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Correct Segmentation Net Training")
    parser.add_argument("--train_dir", dest="train_dir", type=str, required=True)
    parser.add_argument("--test_dir", dest="test_dir", type=str, required=False, default=None)
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
    parser.add_argument(
        "--source",
        dest="source",
        type=str,
        choices=["all", "underseg-all", "overseg-all", "real-underseg"],
        help="The source of the data to train on. Default is to use all data. <underseg-all> means using both synthetic and real underseg datasets; similar for <overseg-all>",
        default="all",
    )
    parser.add_argument(
        "--save_criterions_min",
        dest="save_criterions_min",
        type=str,
        default="test_loss",
        help="The criterions to save the model. The model will be saved if the criterion is the minimum so far. The criterions are separated by comma. The criterion can be one of the following: test_loss, test_acc, ... (see what exists in the validation/test step function)",
    )
    parser.add_argument("--ou_aux", dest="ou_aux", default=False, action="store_true")
    parser.add_argument("--aug-ver", default="v0", type=str, help="The version of the augmentation to use.")
    parser.add_argument("--use-gt-pixel-weight", default=False, action="store_true")
    parser.add_argument("--aux-loss-weight", default=0.5, type=float)

    args = parser.parse_args()

    # convert string to list
    args.aug_scale = [float(x) for x in args.aug_scale.split(",")]
    args.class_weights = [float(x) for x in args.class_weights.split(",")]
    args.save_criterions_min = args.save_criterions_min.split(",")
    return args


def main_train():
    args = parse_args()
    train_csv_filename = "train_data.csv"
    if args.ou_aux:
        train_csv_filename = "train_data_aux.csv"
    print("[Args] ", args)
    # train_dir = Path("./notebook_results/a549_ccp_vim/train_data_v1")
    train_dir = Path(args.train_dir)
    train_csv = train_dir / train_csv_filename
    kernel_size = args.kernel_size
    train_df = pd.read_csv(train_csv)

    print("pd df shape:", train_df.shape)
    print("df samples:", train_df[:2])
    if args.source == "all":
        print(">>> Using all data")
        pass
    elif args.source == "underseg-all":
        print(">>> Using all underseg data")
        # underseg_cols = ["synthetic_underseg_overlap", "real_underseg_cases", "synthetic_underseg_nonoverlap_gauss"]
        indexer = train_df["subdir"].contains("underseg")
        # for col in underseg_cols[1:]:
        #     indexer = indexer | (train_df["subdir"] == col)
        #     assert (train_df["subdir"] == col).sum() > 0, f"no data found in train_df for {col}"
        train_df = train_df[indexer]
        print(">>> after filtering by underseg cases, df shape:", train_df.shape)
    elif args.source == "real-underseg":
        train_df = train_df[train_df["subdir"] == "real_underseg_cases"]
        print(">>> after filtering by real underseg cases, df shape:", train_df.shape)

    # augmentation params
    translation_range = (args.translation, args.translation)
    degrees = args.degrees
    if args.aug_ver == "v0":
        train_transforms = csn_configs.gen_train_transform_v0(degrees, translation_range, args.aug_scale)
    elif args.aug_ver == "v1":
        train_transforms = csn_configs.gen_train_transform_v1(degrees, translation_range, args.aug_scale)
    elif args.aug_ver == "v2":
        train_transforms = csn_configs.gen_train_transform_v2(degrees, translation_range, args.aug_scale)
    elif args.aug_ver == "v3":
        train_transforms = csn_configs.gen_train_transform_v3(degrees, translation_range, args.aug_scale)
    elif args.aug_ver == "v4":
        train_transforms = csn_configs.gen_train_transform_v4(degrees, translation_range, args.aug_scale)
    elif args.aug_ver == "v5":
        train_transforms = csn_configs.gen_train_transform_v5(degrees, translation_range, args.aug_scale)
    elif args.aug_ver == "v6":
        train_transforms = csn_configs.gen_train_transform_v6(degrees, translation_range, args.aug_scale)
    else:
        raise ValueError("Unknown augmentation version")

    def df2dataset(df):
        raw_img_paths = list(df["raw"])
        scaled_seg_mask_paths = list(df["seg"])
        gt_mask_paths = list(df["gt"])
        raw_seg_paths = list(df["raw_seg"])
        scales = list(df["scale"])
        aug_diff_img_paths = list(df["aug_diff_mask"])
        raw_transformed_img_paths = list(df["raw_transformed_img"])
        gt_label_mask_paths = list(df["gt_label_mask"])

        # the source of data: underseg/overseg; real/synthetic
        subdirs = df["subdir"]
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
            raw_df=df,
            subdirs=subdirs,
            use_gt_pixel_weight=args.use_gt_pixel_weight,
        )
        return dataset

    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=args.split_seed)
    if args.debug:
        train_df = train_df[:100]
        val_df = val_df[:10]
    # Get rid of last batch if it's not full to avoid BatchNorm complaints
    train_df = train_df.iloc[: (len(train_df) // args.batch_size) * args.batch_size]
    val_df = val_df.iloc[: (len(val_df) // args.batch_size) * args.batch_size]

    train_dataset = df2dataset(train_df)
    val_dataset = df2dataset(val_df)

    # load test data
    test_dataset = None
    if args.test_dir is not None:
        test_dir = Path(args.test_dir)
        test_csv = test_dir / train_csv_filename
        test_df = pd.read_csv(test_csv)
        test_dataset = df2dataset(test_df)

    logger = TensorBoardLogger(save_dir=".", name="lightning_logs", version=args.model_version)
    if args.debug:
        logger = TensorBoardLogger(save_dir=".", name="test_logs", version=args.model_version)

    if args.ou_aux:
        model = CorrectSegNetAux(
            # train_input_paths=train_input_tuples,
            lr=args.lr,
            num_workers=1,
            batch_size=args.batch_size,
            train_transforms=train_transforms,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            kernel_size=kernel_size,
            loss_type=args.loss,
            class_weights=args.class_weights,
            # only for record keeping purposes; handled by the dataset
            input_type=args.input_type,
            apply_gt_seg_edt=args.apply_gt_seg_edt,
            exclude_raw_input_bg=args.exclude_raw_input_bg,
            aux_loss_weight=args.aux_loss_weight,
        )
    else:
        model = CorrectSegNet(
            # train_input_paths=train_input_tuples,
            lr=args.lr,
            num_workers=1,
            batch_size=args.batch_size,
            train_transforms=train_transforms,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            kernel_size=kernel_size,
            loss_type=args.loss,
            class_weights=args.class_weights,
            # only for record keeping purposes; handled by the dataset
            input_type=args.input_type,
            apply_gt_seg_edt=args.apply_gt_seg_edt,
            exclude_raw_input_bg=args.exclude_raw_input_bg,
        )

    print("logger save dir:", logger.save_dir)
    print("logger subdir:", logger.sub_dir)
    print("logger version:", logger.version)
    # best_models_checkpoint_callback = ModelCheckpoint(
    #     save_top_k=10,
    #     monitor="test_loss_real_underseg_cases",
    #     mode="min",
    #     filename="{epoch:02d}-{test_loss_real_underseg_cases:.4f}",
    # )
    # best_models_checkpoint_callback = ModelCheckpoint(
    #     save_top_k=10,
    #     monitor="test_out_matched_num_gt_iou_0.5_percent_real_underseg_cases",
    #     mode="max",
    #     filename="{epoch:02d}-{test_out_matched_num_gt_iou_0.5_percent_real_underseg_cases:.4f}",
    # )
    # ckpt_callbacks = [best_models_checkpoint_callback, last_models_checkpoint_callback]

    ckpt_callbacks = []
    for criterion in args.save_criterions_min:
        ckpt_callbacks.append(
            ModelCheckpoint(
                save_top_k=3,
                monitor=criterion,
                mode="min",
                filename="{epoch:02d}-{" + criterion + ":.4f}",
            )
        )
    last_models_checkpoint_callback = ModelCheckpoint(
        save_last=True,
        filename="{epoch}-{global_step}",
    )
    ckpt_callbacks.append(last_models_checkpoint_callback)

    trainer = Trainer(
        gpus=1,
        max_epochs=args.epochs,
        resume_from_checkpoint=args.model_ckpt,
        logger=logger,
        callbacks=ckpt_callbacks,
        log_every_n_steps=100,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main_train()
