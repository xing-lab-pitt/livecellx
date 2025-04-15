import json
import torch
import numpy as np
import random
from pathlib import Path
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux
from livecellx.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset
from livecellx.model_zoo.segmentation.eval_csn import compute_metrics, assemble_train_test_dataset
import livecellx.model_zoo.segmentation.csn_configs as csn_configs
from livecellx.model_zoo.segmentation import custom_transforms
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Correct Segmentation Net Training with Active Learning")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--model_version", type=str, default=None)
    parser.add_argument("--kernel_size", type=int, default=1)
    parser.add_argument("--backbone", type=str, choices=["deeplabV3", "unet_aux"], default="deeplabV3")
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--translation", type=float, default=0.5)
    parser.add_argument("--degrees", type=int, default=180)
    parser.add_argument("--aug_scale", type=str, default="0.5,1.5")
    parser.add_argument("--split_seed", type=int, default=237)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--input_type", type=str, default="raw_aug_seg")
    parser.add_argument("--apply_gt_seg_edt", default=False, action="store_true")
    parser.add_argument("--class-weights", type=str, default="1,1,1")
    parser.add_argument("--loss", type=str, default="CE")
    parser.add_argument("--exclude_raw_input_bg", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--source", type=str, default="all")
    parser.add_argument("--save_criterions_min", type=str, default="test_loss")
    parser.add_argument("--ou_aux", default=False, action="store_true")
    parser.add_argument("--aug-ver", default="v0", type=str)
    parser.add_argument("--use-gt-pixel-weight", default=False, action="store_true")
    parser.add_argument("--aux-loss-weight", default=0.5, type=float)
    parser.add_argument("--normalize_uint8", default=False, action="store_true")
    parser.add_argument("--torch_seed", default=237, type=int)
    parser.add_argument("--normalize_gt_edt", default=False, action="store_true")
    parser.add_argument("--out_threshold", type=float, default=0.6)
    parser.add_argument(
        "--best_model_ckpt_path",
        type=str,
        default="./lightning_logs/version_v18_02-inEDTv1-augEdtV9-scaleV2-lr-0.0001-aux-seed-404/checkpoints/epoch=453-global_step=0.ckpt",
    )
    parser.add_argument(
        "--test_pert",
        default=0.1,
        type=float,
    )
    args = parser.parse_args()
    args.aug_scale = [float(x) for x in args.aug_scale.split(",")]
    args.class_weights = [float(x) for x in args.class_weights.split(",")]
    args.save_criterions_min = args.save_criterions_min.split(",")
    return args


def binary_entropy(prob):
    ent = -torch.sum(
        torch.mul(prob, torch.log2(prob + 1e-30)) + torch.mul(1.0 - prob, torch.log2(1.0 - prob + 1e-30))
    ) / torch.count_nonzero(prob)
    return ent


def rank_unlabeled_data(objectives, senses=None):
    from paretoset import paretorank

    if senses is None:
        senses = ["max"] * objectives.shape[1]
    return paretorank(objectives, senses)


def df2dataset(df, transforms, args):
    dataset = CorrectSegNetDataset(
        list(df["raw"]),
        list(df["seg"]),
        list(df["gt"]),
        gt_label_mask_paths=list(df["gt_label_mask"]),
        raw_seg_paths=list(df["raw_seg"]),
        scales=list(df["scale"]),
        transform=transforms,
        raw_transformed_img_paths=list(df["raw_transformed_img"]),
        aug_diff_img_paths=list(df["aug_diff_mask"]),
        input_type=args.input_type,
        apply_gt_seg_edt=args.apply_gt_seg_edt,
        exclude_raw_input_bg=args.exclude_raw_input_bg,
        raw_df=df,
        subdirs=df["subdir"],
        use_gt_pixel_weight=args.use_gt_pixel_weight,
        normalize_gt_edt=args.normalize_gt_edt,
    )
    return dataset


def run_active_learning(args, train_df, val_df, test_df, train_transforms, iteration=100, quota_per_iteration=512):
    labeled_data_idx = np.zeros(len(train_df)).astype(bool)
    init_labeled_idx = list(range(len(train_df)))
    random.shuffle(init_labeled_idx)
    labeled_data_idx[init_labeled_idx[:quota_per_iteration]] = True
    unlabeled_data_idx = ~labeled_data_idx

    logger = TensorBoardLogger(save_dir=".", name="lightning_logs_AL", version=args.model_version)
    # Get directory path from logger
    iter_metrics = {
        "train_labeled": [],
        "test_labeled": [],
        "train_all": [],
        "test_all": [],
        "train_best_model_labeled": [],
        "test_best_model_labeled": [],
    }
    iter_train_labeled_dfs = []
    model = CorrectSegNetAux(
        lr=args.lr,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        train_transforms=train_transforms,
        train_dataset=df2dataset(train_df[labeled_data_idx], train_transforms, args),
        val_dataset=df2dataset(val_df, train_transforms, args),
        test_dataset=df2dataset(test_df, train_transforms, args),
        kernel_size=args.kernel_size,
        loss_type=args.loss,
        class_weights=args.class_weights,
        input_type=args.input_type,
        apply_gt_seg_edt=args.apply_gt_seg_edt,
        exclude_raw_input_bg=args.exclude_raw_input_bg,
        aux_loss_weight=args.aux_loss_weight,
        backbone=args.backbone,
        normalize_uint8=args.normalize_uint8,
    )
    model.cuda()
    model_best = CorrectSegNetAux.load_from_checkpoint(
        args.best_model_ckpt_path,
    )
    model_best.cuda().eval()

    trainer = Trainer(
        gpus=1,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[
            ModelCheckpoint(save_top_k=3, monitor="val_loss", mode="min", filename="{epoch:02d}-{val_loss:.4f}"),
            ModelCheckpoint(save_last=True, filename="{epoch}-{global_step}"),
        ],
        log_every_n_steps=100,
        check_val_every_n_epoch=50,
        val_check_interval=5,
    )

    logger_dir = Path(logger.log_dir)
    eval_transform = csn_configs.CustomTransformEdtV9(use_gaussian_blur=True, gaussian_blur_sigma=30)
    for iter_num in range(iteration):
        print(f"[AL] Iteration {iter_num+1}, start training...")
        model.cuda().train()
        model.train_dataset = df2dataset(train_df[labeled_data_idx], train_transforms, args)
        trainer.fit(model)
        iter_train_labeled_dfs.append(train_df[labeled_data_idx].copy())
        iter_train_labeled_dfs[-1]["iter"] = iter_num

        print("[AL EVAL] evaluating model...")
        model.cuda().eval()
        (
            train_label_dataset_eval,
            _,
            test_label_dataset_eval,
        ) = assemble_train_test_dataset(train_df[labeled_data_idx], test_df, model, train_split=1)
        train_label_dataset_eval.transform = eval_transform
        test_label_dataset_eval.transform = eval_transform
        output_train_label_eval = compute_metrics(
            train_label_dataset_eval, model, out_threshold=args.out_threshold, return_mean=True
        )
        # print("[AL EVAL] flag1 evaluating model on test label set...")
        output_test_label_eval = compute_metrics(
            test_label_dataset_eval, model, out_threshold=args.out_threshold, return_mean=True
        )
        # train_all_dataset_eval, _, test_all_dataset_eval, = assemble_train_test_dataset(
        #     train_df, test_df, model, train_split=1
        # )
        # train_all_dataset_eval.transform = eval_transform
        # test_all_dataset_eval.transform = eval_transform
        # output_train_all_eval = compute_metrics(
        #     train_all_dataset_eval, model, out_threshold=args.out_threshold, return_mean=True
        # )
        # output_test_all_eval = compute_metrics(
        #     test_all_dataset_eval, model, out_threshold=args.out_threshold, return_mean=True
        # )

        output_train_best_label_eval = compute_metrics(
            train_label_dataset_eval, model_best, out_threshold=args.out_threshold, return_mean=True
        )

        output_test_best_label_eval = compute_metrics(
            test_label_dataset_eval, model_best, out_threshold=args.out_threshold, return_mean=True
        )
        iter_metrics["train_labeled"].append(output_train_label_eval)
        iter_metrics["test_labeled"].append(output_test_label_eval)
        # iter_metrics["test_all"].append(output_test_all_eval)
        # iter_metrics["test_all"].append(output_test_all_eval)
        iter_metrics["train_best_model_labeled"].append(output_train_best_label_eval)
        iter_metrics["test_best_model_labeled"].append(output_test_best_label_eval)

        # Output iter metrics to JSON
        with open(logger_dir / f"iter_metrics.json", "w+") as f:
            json.dump(iter_metrics, f)
        # Output iter train label dfs, add a column "iter" to represent the iteration number
        combined_df = pd.concat(iter_train_labeled_dfs)
        combined_df.to_csv(logger_dir / f"iter_train_labeled_combined_df.csv", index=False)

        unlabeled_dataset = df2dataset(train_df[unlabeled_data_idx], train_transforms, args)
        if len(unlabeled_dataset) == 0:
            print("[AL] No unlabeled data left, exiting...")
            break
        scores = []
        for sample in unlabeled_dataset:
            x = sample["input"].unsqueeze(0).cuda()
            with torch.no_grad():
                seg_out, aux_out = model(x)
                seg_out = seg_out.detach().cpu()
                aux_out = aux_out.detach().cpu()
            logits = model.output_to_logits(seg_out)[0]
            entropies = [binary_entropy(logits[i]) for i in range(logits.shape[0])]
            scores.append(entropies)

        ranking = rank_unlabeled_data(np.array(scores))
        unlabeled_data_idx_idx = np.where(unlabeled_data_idx)[0]
        selected = unlabeled_data_idx_idx[np.argsort(ranking)[:quota_per_iteration]]

        labeled_data_idx[selected] = True
        unlabeled_data_idx = ~labeled_data_idx


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.torch_seed)
    train_csv_filename = "train_data.csv"
    if args.ou_aux:
        train_csv_filename = "train_data_aux.csv"
    else:
        assert False, "Latest CS-Net expects auxiliary loss..."

    train_dir = Path(args.train_dir)
    train_csv = train_dir / train_csv_filename
    train_df = pd.read_csv(train_csv)

    if args.source == "underseg-all":
        train_df = train_df[train_df["subdir"].str.contains("underseg")]
    elif args.source == "real-underseg":
        train_df = train_df[train_df["subdir"] == "real_underseg_cases"]
    elif args.source == "dropout":
        train_df = train_df[train_df["subdir"].str.contains("dropout")]

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=args.split_seed)
    if args.debug:
        train_df = train_df[:100]
        val_df = val_df[:100]
        args.epochs = 1
    train_df = train_df.iloc[: (len(train_df) // args.batch_size) * args.batch_size]
    val_df = val_df.iloc[: (len(val_df) // args.batch_size) * args.batch_size]

    test_df = None
    if args.test_dir is not None:
        test_dir = Path(args.test_dir)
        test_csv = test_dir / train_csv_filename
        test_df = pd.read_csv(test_csv)

    test_df = test_df[:int(len(test_df) * args.test_pert)]
    if args.debug:
        test_df = test_df[:100]

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
    elif args.aug_ver == "v7":
        train_transforms = csn_configs.gen_train_transform_v7(degrees, translation_range, args.aug_scale)
    elif args.aug_ver == "edt-v8":
        train_transforms = csn_configs.gen_train_transform_edt_v8(
            degrees, translation_range, args.aug_scale, shear=10, flip_p=0.5
        )
    elif args.aug_ver == "edt-v9":
        train_transforms = custom_transforms.CustomTransformEdtV9(
            degrees=degrees,
            translation_range=translation_range,
            scale=args.aug_scale,
            shear=10,
            flip_p=0.5,
            use_gaussian_blur=True,
        )
    else:
        raise ValueError("Unknown augmentation version")

    run_active_learning(args, train_df, val_df, test_df, train_transforms)
