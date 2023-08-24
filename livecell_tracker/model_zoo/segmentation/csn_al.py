import torch


import sys

sys.path.append("./../")
sys.path.append("./../../")
sys.path.append("./../../../")
import numpy as np

torch.manual_seed(237)
import argparse
from pathlib import Path
import pandas as pd
from torchvision import transforms
import torch
import torch.utils.data
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from livecell_tracker.model_zoo.segmentation.sc_correction import CorrectSegNet
from livecell_tracker.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset
from livecell_tracker.model_zoo.segmentation.eval_csn import compute_metrics, assemble_train_test_dataset


# python ./csn_al.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v11" --test_dir="./notebook_results/a549_ccp_vim/test_data_v11"  --source=all --model_version=version_$model --epochs=1 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --loss=BCE
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
    parser.add_argument(
        "--out_threshold",
        type=float,
        default=0.6,
        help="threshold for output mask; [0, 1] for binary logits prediction, >=1 for edt regression prediction",
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
    args = parser.parse_args()

    # convert string to list
    args.aug_scale = [float(x) for x in args.aug_scale.split(",")]
    args.class_weights = [float(x) for x in args.class_weights.split(",")]
    args.save_criterions_min = args.save_criterions_min.split(",")
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
    if args.source == "all":
        print(">>> Using all data")
        pass
    elif args.source == "underseg-all":
        print(">>> Using all underseg data")
        underseg_cols = ["synthetic_underseg_overlap", "real_underseg_cases", "synthetic_underseg_nonoverlap_gauss"]
        indexer = train_df["subdir"] == underseg_cols[0]
        for col in underseg_cols[1:]:
            indexer = indexer | (train_df["subdir"] == col)
            assert (train_df["subdir"] == col).sum() > 0, f"no data found in train_df for {col}"

        train_df = train_df[indexer]
        print(">>> after filtering by underseg cases, df shape:", train_df.shape)
    elif args.source == "real-underseg":
        train_df = train_df[train_df["subdir"] == "real_underseg_cases"]
        print(">>> after filtering by real underseg cases, df shape:", train_df.shape)

    # augmentation params
    translation_range = (args.translation, args.translation)
    degrees = args.degrees
    train_transforms = transforms.Compose(
        [
            # transforms.Resize((412, 412)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=degrees, translate=translation_range, scale=args.aug_scale),
            transforms.RandomCrop((412, 412), pad_if_needed=True),
        ]
    )

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
        )
        return dataset

    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=args.split_seed)
    if args.debug:
        train_df = train_df[:100]
        val_df = val_df[:10]

    train_dataset = df2dataset(train_df)
    val_dataset = df2dataset(val_df)

    # load test data
    test_dataset = None
    if args.test_dir is not None:
        test_dir = Path(args.test_dir)
        test_csv = test_dir / "train_data.csv"
        test_df = pd.read_csv(test_csv)
        test_dataset = df2dataset(test_df)

    logger = TensorBoardLogger(save_dir=".", name="lightning_logs", version=args.model_version)
    if args.debug:
        logger = TensorBoardLogger(save_dir=".", name="test_logs", version=args.model_version)

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
        progress_bar_refresh_rate=0,
        log_every_n_steps=100,
    )

    iteration = 10
    iter_num = 0
    ckpt_path = "./lightning_logs/version_v11_01/checkpoints/last.ckpt"
    model_best = CorrectSegNet.load_from_checkpoint(
        ckpt_path,
    )

    labeled_data_idx = []
    unlabeled_data_idx = []
    quota_per_iteration = 100

    # set init labeled set 100
    import random

    labeled_data_idx = np.zeros(len(train_df)).astype(bool)
    init_labeled_idx = list(range(len(train_df)))
    random.shuffle(init_labeled_idx)
    init_labeled_idx = init_labeled_idx[:quota_per_iteration]
    labeled_data_idx[init_labeled_idx] = True
    print(labeled_data_idx)
    unlabeled_data_idx = ~labeled_data_idx

    while iter_num < iteration:
        iter_num = iter_num + 1
        model.train()
        model.train_dataset = df2dataset(train_df[labeled_data_idx])
        unlabeled_dataset = df2dataset(train_df[unlabeled_data_idx])
        trainer.fit(model)
        model.cuda()

        train_dataset, val_dataset, test_dataset, whole_dataset = assemble_train_test_dataset(train_df, test_df, model)
        # train_dataset.cuda()
        # whole_dataset.cuda()
        # print('train_dataset', train_dataset)
        model.eval()
        model_best.cuda()
        model_best.eval()
        train_dataset_eval, val_dataset_eval, test_dataset_eval, whole_dataset_eval = assemble_train_test_dataset(
            train_df[labeled_data_idx], test_df, model
        )
        output_train_eval = compute_metrics(
            train_dataset_eval, model, out_threshold=args.out_threshold, whole_dataset=whole_dataset_eval
        )
        output_test_eval = compute_metrics(
            test_dataset_eval, model, out_threshold=args.out_threshold, whole_dataset=whole_dataset_eval
        )
        # save metrics
        print("[EVAL] saving metrics")

        avg_train_metrics = {key: np.mean(value) for key, value in output_train_eval.items()}
        avg_test_metrics = {key: np.mean(value) for key, value in output_test_eval.items()}
        print(avg_train_metrics)
        print(avg_test_metrics)

        unlabeled_data_idx_idx = np.array(np.where(unlabeled_data_idx == True))[0]
        print(unlabeled_data_idx_idx.shape)
        scores = []
        for sample in unlabeled_dataset:
            prediction = model(sample["input"].unsqueeze(0).cuda()).detach().cpu()
            prediction_logit = model.output_to_logits(prediction)[0]

            ent_0 = binary_entropy(prediction_logit[0, :, :])
            ent_1 = binary_entropy(prediction_logit[1, :, :])
            ent_2 = binary_entropy(prediction_logit[2, :, :])
            scores.append([ent_0, ent_1, ent_2])

        scores = np.array(scores)
        import collections

        ranking = rank_unlabeled_data(scores)
        # print(collections.Counter(ranking))
        selection = unlabeled_data_idx_idx[(1.0 * ranking).argsort()[:quota_per_iteration]]

        # update labeled & unlabeled data
        labeled_data_idx[selection] = True
        unlabeled_data_idx = ~labeled_data_idx
    # trainer.fit(model)


# new algorithm added here


def al_query_func(unlabeled_data_idx, labeled_data_idx, data, model):
    # prob of unlabeled data
    unlabeled_data = df2dataset(data[unlabeled_data_idx])
    labeled_data = df2dataset(data[labeled_data_idx])
    output = compute_metrics(unlabeled_data, model, out_threshold=args.out_threshold, whole_dataset=None)
    normal_prob, under_prob, over_prob = output[:, 0, :, :], output[:, 1, :, :], output[:, 2, :, :]
    print(normal_prob[:2])
    print(under_prob[:2])
    print(over_prob[:2])


def binary_entropy(prob):
    # ent = - torch.sum(torch.mul(prob, torch.log2(prob+1e-30)) + torch.mul(1.0-prob, torch.log2(1.0 - prob+1e-30))) / (prob.shape[0]*prob.shape[1])
    ent = -torch.sum(
        torch.mul(prob, torch.log2(prob + 1e-30)) + torch.mul(1.0 - prob, torch.log2(1.0 - prob + 1e-30))
    ) / torch.count_nonzero(prob)
    return ent


def rank_unlabeled_data(objectives, senses=None):
    from paretoset import paretoset, paretorank

    # the objectives are defaulted as max
    if senses is None:
        senses = ["max"] * objectives.shape[1]
    return paretorank(objectives, senses)


if __name__ == "__main__":
    main_train()
