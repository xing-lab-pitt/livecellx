import argparse
import glob
from pathlib import Path
from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.measure

import torch
import torch
import torch.utils.data
import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
import pandas as pd
import os
import time
import random
from livecellx.model_zoo.segmentation.sc_correction import CorrectSegNet
from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux
from livecellx.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset


def assemble_dataset(
    df: pd.DataFrame, apply_gt_seg_edt=False, exclude_raw_input_bg=False, input_type=None, use_gt_pixel_weight=False
):
    assert input_type is not None
    raw_img_paths = list(df["raw"])
    scaled_seg_mask_paths = list(df["seg"])
    gt_mask_paths = list(df["gt"])
    raw_seg_paths = list(df["raw_seg"])
    scales = list(df["scale"])
    aug_diff_img_paths = list(df["aug_diff_mask"])
    raw_transformed_img_paths = list(df["raw_transformed_img"])
    gt_label_mask_paths = list(df["gt_label_mask"])

    dataset = CorrectSegNetDataset(
        raw_img_paths,
        scaled_seg_mask_paths,
        gt_mask_paths,
        gt_label_mask_paths=gt_label_mask_paths,
        raw_seg_paths=raw_seg_paths,
        scales=scales,
        transform=None,
        raw_transformed_img_paths=raw_transformed_img_paths,
        aug_diff_img_paths=aug_diff_img_paths,
        apply_gt_seg_edt=apply_gt_seg_edt,
        exclude_raw_input_bg=exclude_raw_input_bg,
        input_type=input_type,
        raw_df=df,
        use_gt_pixel_weight=use_gt_pixel_weight,
    )
    return dataset


def assemble_train_test_dataset(train_df, test_df, model, split_seed=237):  # default seed used in our CSN paper

    dataset = assemble_dataset(
        train_df,
        apply_gt_seg_edt=model.apply_gt_seg_edt,
        exclude_raw_input_bg=model.exclude_raw_input_bg,
        input_type=model.input_type,
    )
    train_sample_num = int(len(dataset) * 0.8)
    val_sample_num = len(dataset) - train_sample_num
    split_generator = torch.Generator().manual_seed(split_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_sample_num, val_sample_num], generator=split_generator
    )

    test_dataset = assemble_dataset(
        test_df,
        apply_gt_seg_edt=model.apply_gt_seg_edt,
        exclude_raw_input_bg=model.exclude_raw_input_bg,
        input_type=model.input_type,
    )
    return train_dataset, val_dataset, test_dataset, dataset


def match_label_mask_by_iou(out_label_mask, gt_label_mask, bg_label=0, match_threshold=0.8, return_iou_list=True):
    assert out_label_mask.shape == gt_label_mask.shape

    out_labels = np.unique(out_label_mask)
    gt_labels = np.unique(gt_label_mask)

    # remove bg_label
    out_labels = out_labels[out_labels != bg_label]
    gt_labels = gt_labels[gt_labels != bg_label]

    # calculate iou mapping between out labels and gt labels
    label_gt2out = {gt: [] for gt in gt_labels}
    label_out2gt = {out: [] for out in out_labels}
    gt_out_iou_list = []
    for gt_label in gt_labels:
        gt_mask = gt_label_mask == gt_label
        label_gt2out[gt_label] = []

        for out_label in out_labels:
            out_mask = out_label_mask == out_label
            iou = np.sum(out_mask & gt_mask) / np.sum(out_mask | gt_mask)
            gt_out_iou_list.append((gt_label, out_label, iou))
            if iou > match_threshold:
                label_gt2out[gt_label].append(out_label)
                label_out2gt[out_label].append(gt_label)

    # because it is a 2D imaging label mapping, there is no overlapping and only one region may cross sufficiently high iou thresholds
    matched_num = 0
    for gt_label in label_gt2out:
        if len(label_gt2out[gt_label]) != 1:
            continue
        matched_num += 1
    gt_num = len(gt_labels)
    out_num = len(out_labels)
    if return_iou_list:
        return matched_num, out_num, gt_num, np.array(gt_out_iou_list)
    return matched_num, out_num, gt_num


def evaluate_sample_v3_underseg(
    sample: dict,
    model: CorrectSegNet,
    raw_seg=None,
    scale=None,
    out_threshold=0.6,
    gt_label_mask=None,
    gt_iou_match_thresholds=[0.5, 0.8, 0.9, 0.95],  # eval on a range of thresholds
):
    assert len(gt_iou_match_thresholds) > 0
    out_mask = model(sample["input"].unsqueeze(0).cuda())
    if isinstance(model, CorrectSegNetAux):
        seg_out_mask = out_mask[0]
        aux_out = out_mask[1]
    elif isinstance(model, CorrectSegNet):
        seg_out_mask = out_mask
    else:
        raise ValueError("model type not supported")

    if model.loss_type == "BCE" or model.loss_type == "CE":
        seg_out_mask = model.output_to_logits(seg_out_mask)

    # 1 sample -> get first batch
    seg_out_mask = seg_out_mask.cpu().detach().numpy().squeeze()

    # seg_out_mask = skimage.transform.resize(seg_out_mask, sample["gt_mask"].shape[1:], order=1, mode="reflect")
    assert seg_out_mask.shape[0] == 3, "Expected 3 channels for seg_out_mask, got shape=%s" % str(seg_out_mask.shape)

    original_input_mask = sample["seg_mask"].numpy().squeeze()
    original_input_mask = original_input_mask.astype(bool)
    original_label_mask = skimage.measure.label(original_input_mask)

    gt_seg_mask = sample["gt_mask_binary"].numpy().squeeze().astype(bool)

    original_cell_count = len(np.unique(original_label_mask)) - 1  # -1 for bg

    assert gt_label_mask is not None, "gt_label_mask is required for evaluation"
    assert (
        len(set(np.unique(gt_seg_mask).tolist())) <= 2
    ), "More than two labels in the gt masks. Please remove this assertation if you are working on mapping cases with more than 2 gt cells (the case in LCA paper)."

    combined_over_under_seg = np.zeros([3] + list(seg_out_mask.shape[1:]))
    combined_over_under_seg[0, seg_out_mask[1, :] > out_threshold] = 1
    combined_over_under_seg[1, seg_out_mask[2, :] > out_threshold] = 1

    # ignore pixels outside an area, only works for undersegmentation
    out_mask_predicted = seg_out_mask[0] > out_threshold
    # TODO: the following line does not hold for overseg case; double check with the team
    # out_mask_predicted[original_input_mask < 0.5] = 0
    out_mask_predicted = out_mask_predicted.astype(bool)

    # match gt label mask with out label mask
    out_label_mask = skimage.measure.label(out_mask_predicted)

    # Resize out_label_mask to match gt_label_mask (example: (412, 412) to (136, 172)) if GT label is not enhanced during dataset preparation
    # [TODO] Remvoe if label masks are augmented during dataset preparation
    out_label_mask = skimage.transform.resize(out_label_mask, gt_label_mask.shape, order=0, mode="reflect")
    original_label_mask = skimage.transform.resize(original_label_mask, gt_label_mask.shape, order=0, mode="reflect")

    out_matched_num, out_cell_count, gt_cell_num, gt_out_iou_list = match_label_mask_by_iou(
        out_label_mask, gt_label_mask, match_threshold=gt_iou_match_thresholds[0], return_iou_list=True
    )
    origin_matched_num, origin_cell_count, gt_cell_num, gt_origin_iou_list = match_label_mask_by_iou(
        original_label_mask, gt_label_mask, return_iou_list=True
    )

    metrics_dict = {}
    metrics_dict["out_mask_accuracy"] = (out_mask_predicted == gt_seg_mask).sum() / np.prod(out_mask_predicted.shape)
    metrics_dict["original_mask_accuracy"] = (original_input_mask == gt_seg_mask).sum() / np.prod(
        out_mask_predicted.shape
    )
    metrics_dict["out_mask_iou"] = (out_mask_predicted & gt_seg_mask).sum() / (out_mask_predicted | gt_seg_mask).sum()
    metrics_dict["original_mask_iou"] = (original_input_mask & gt_seg_mask).sum() / (
        original_input_mask | gt_seg_mask
    ).sum()

    if gt_cell_num == 0:
        print(">>> Warning: no gt cells in this sample and thus gt_cell_num is 0.")
        gt_cell_num = np.inf

    metrics_dict["out_cell_count"] = out_cell_count
    metrics_dict["gt_cell_count"] = gt_cell_num
    metrics_dict["out_minus_gt_count"] = out_cell_count - gt_cell_num
    metrics_dict["abs_out_count_diff"] = abs(gt_cell_num - out_cell_count)
    metrics_dict["abs_out_count_diff_percent"] = abs(gt_cell_num - out_cell_count) / gt_cell_num
    metrics_dict["abs_original_count_diff"] = abs(gt_cell_num - original_cell_count)
    metrics_dict["matched_num"] = out_matched_num

    for threshold in gt_iou_match_thresholds:
        _matched_num = gt_out_iou_list[:, 2] > threshold if len(gt_out_iou_list) > 0 else np.array([0, 0])
        metrics_dict[f"out_matched_num_gt_iou_{threshold}"] = _matched_num.sum()
        metrics_dict[f"out_matched_num_gt_iou_{threshold}_percent"] = _matched_num.sum() / gt_cell_num
        metrics_dict[f"out_matched_num_gt_iou_total_match"] = _matched_num.sum() == gt_cell_num

    for threshold in gt_iou_match_thresholds:
        _matched_num = gt_origin_iou_list[:, 2] > threshold if len(gt_origin_iou_list) > 0 else np.array([0, 0])
        metrics_dict[f"origin_matched_num_gt_origin_{threshold}"] = _matched_num.sum()
        metrics_dict[f"origin_matched_num_gt_origin_{threshold}_percent"] = _matched_num.sum() / gt_cell_num
        metrics_dict[f"origin_matched_num_gt_iou_total_match"] = _matched_num.sum() == gt_cell_num

    # metrics_dict["gt_iou_match_threshold"] = gt_iou_match_threshold
    # metrics_dict["gt_out_iou_list"] = gt_out_iou_list

    # Check if aux out is correct
    if isinstance(model, CorrectSegNetAux):
        aux_out_class = aux_out.squeeze().argmax().item()
        aux_vec = np.zeros(4)
        aux_vec[aux_out_class] = 1
        if list(sample["ou_aux"]) != list(aux_vec):
            metrics_dict["aux_out_correct"] = 0
        else:
            metrics_dict["aux_out_correct"] = 1
    return metrics_dict


def compute_metrics(
    dataset: Union[CorrectSegNetDataset, torch.utils.data.Subset],
    model,
    out_threshold=0.6,
    whole_dataset: CorrectSegNetDataset = None,
    gt_label_masks: List[np.ndarray] = None,
):
    res_metrics = {}
    for i, sample in enumerate(tqdm.tqdm(dataset)):
        if gt_label_masks is not None:
            gt_label_mask = gt_label_masks[i]
        elif isinstance(dataset, torch.utils.data.Subset):
            assert whole_dataset is not None, "whole_dataset must be provided when <dataset> function arg is a Subset"
            origin_idx = dataset.indices[i]
            gt_label_mask = whole_dataset.get_gt_label_mask(origin_idx)
        else:
            gt_label_mask = dataset.get_gt_label_mask(i)

        single_sample_metrics = evaluate_sample_v3_underseg(
            sample, model, out_threshold=out_threshold, gt_label_mask=gt_label_mask
        )
        for metric, value in single_sample_metrics.items():
            if metric not in res_metrics:
                res_metrics[metric] = []
            res_metrics[metric].append(value)

    for key in res_metrics:
        res_metrics[key] = np.array(res_metrics[key])
    return res_metrics


def viz_sample_v3(sample: dict, model, raw_seg=None, scale=None, out_threshold=0.6, save_path=None, close_on_save=True):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def add_colorbar(im, ax, fig):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

    out_mask = model(sample["input"].unsqueeze(0).cuda())
    original_input_mask = sample["input"].numpy().squeeze()[2]
    original_input_mask = original_input_mask.astype(bool)

    gt_mask = sample["gt_mask"].numpy().squeeze()
    out_mask = out_mask[0]
    if model.loss_type == "CE" or model.loss_type == "BCE":
        out_mask = model.output_to_logits(out_mask)
    out_mask = out_mask.cpu().detach().numpy()
    fig, axes = plt.subplots(1, 12, figsize=(12 * 7, 6))

    ax_idx = 0
    ax = axes[ax_idx]
    ax.imshow(sample["input"][0])
    ax.set_title("input: dim0")

    ax_idx += 1
    ax = axes[ax_idx]
    ax.imshow(sample["input"][1])
    ax.set_title("input: dim1")

    ax_idx += 1
    ax = axes[ax_idx]
    ax.imshow(sample["input"][2])
    ax.set_title("input:dim2")

    ax_idx += 1
    ax = axes[ax_idx]
    im2 = ax.imshow(out_mask[0, :])
    ax.set_title("out0seg")
    add_colorbar(im2, ax, fig)

    ax_idx += 1
    ax = axes[ax_idx]
    ax.imshow(gt_mask[0, :])
    ax.set_title("gt0 seg")

    ax_idx += 1
    ax = axes[ax_idx]
    im4 = ax.imshow(out_mask[1, :])
    ax.set_title("out1seg")
    add_colorbar(im4, ax, fig)

    ax_idx += 1
    ax = axes[ax_idx]
    im5 = ax.imshow(gt_mask[1, :])
    add_colorbar(im5, ax, fig)
    ax.set_title("gt1 seg")

    ax_idx += 1
    ax = axes[ax_idx]
    im6 = ax.imshow(out_mask[2, :])
    add_colorbar(im6, ax, fig)
    ax.set_title("out2 seg")

    ax_idx += 1
    ax = axes[ax_idx]
    im7 = ax.imshow(gt_mask[2, :])
    add_colorbar(im7, ax, fig)
    ax.set_title("gt2 seg")

    combined_over_under_seg = np.zeros([3] + list(out_mask.shape[1:]))
    combined_over_under_seg[0, out_mask[1, :] > out_threshold] = 1
    combined_over_under_seg[1, out_mask[2, :] > out_threshold] = 1

    ax_idx += 1
    ax = axes[ax_idx]
    im = ax.imshow(np.moveaxis(combined_over_under_seg, 0, 2))
    ax.set_title("out(1,2), over/under seg combined")

    # import matplotlib.patches as mpatches
    # values = [-1, 0, 1]
    # colors = [im.cmap(im.norm(value)) for value in values]
    # patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values))]
    # ax.legend(handles=patches, loc=2, borderaxespad=0. )
    ax_idx += 1
    ax = axes[ax_idx]
    ax.imshow(out_mask[0] > out_threshold)
    ax.set_title(f"out0 >{out_threshold} threshold")

    out_mask_predicted = out_mask[0] > out_threshold
    # ignore pixels outside an area, only works for undersegmentation
    # out_mask_predicted[original_input_mask < 0.5] = 0
    out_mask_predicted = out_mask_predicted.astype(bool)

    ax_idx += 1
    ax = axes[ax_idx]
    ax.imshow(out_mask_predicted)
    ax.set_title(f"cleaned out mask prediction")

    if save_path is not None:
        plt.savefig(save_path)
    if save_path and close_on_save:
        plt.close()


def viz_sample_only(sample):
    fig, axes = plt.subplots(1, 7, figsize=(3 * 7, 6))
    ax_idx = 0
    ax = axes[ax_idx]
    ax.imshow(sample["input"][0])
    ax.set_title("input: dim0")

    ax_idx += 1
    ax = axes[ax_idx]
    ax.imshow(sample["input"][1])
    ax.set_title("input: dim1")

    ax_idx += 1
    ax = axes[ax_idx]
    ax.imshow(sample["input"][2])
    ax.set_title("input:dim2")

    # gt
    ax_idx += 1
    ax = axes[ax_idx]
    ax.imshow(sample["gt_mask"][0])
    ax.set_title("gt0 seg")

    ax_idx += 1
    ax = axes[ax_idx]
    ax.imshow(sample["gt_mask"][1])
    ax.set_title("gt1 seg")

    ax_idx += 1
    ax = axes[ax_idx]
    ax.imshow(sample["gt_mask"][2])
    ax.set_title("gt2 seg")

    ax_idx += 1
    ax = axes[ax_idx]
    ax.imshow(sample["gt_label_mask"])
    ax.set_title("gt label mask")


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct Segmentation Net Evaluation")
    parser.add_argument("--name", type=str, help="name of the evaluation", required=True)
    parser.add_argument("--ckpt", dest="ckpt", type=str, help="path to model checkpoint", default=None)
    parser.add_argument("--train_dir", type=str, help="./notebook_results/a549_ccp_vim/train_data_v4/")
    parser.add_argument("--test_dir", type=str, help="./notebook_results/a549_ccp_vim/test_data_v4/")
    parser.add_argument(
        "--pl_dir", type=str, help="a directory containing  ./checkpoints/epoch=xxxx-step=xxxx.ckpt", default=None
    )
    parser.add_argument(
        "--out_threshold",
        type=float,
        default=0.6,
        help="threshold for output mask; [0, 1] for binary logits prediction, >=1 for edt regression prediction",
    )
    parser.add_argument("--save_dir", type=str, default="./eval_results/")
    parser.add_argument("--debug", dest="debug", default=False, action="store_true")
    parser.add_argument("--wait_for_gpu_mem", dest="wait_for_gpu_mem", default=False, action="store_true")
    parser.add_argument(
        "--viz_pred",
        dest="viz_pred",
        default=False,
        action="store_true",
        help="visualize predictions and raw images, save to save_dir",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    args = parser.parse_args()
    return args


def get_cuda_free_memory():
    """Get the free memory of the GPU in bytes"""
    global_free, occupied = torch.cuda.mem_get_info()
    print("free memory: %fGB" % (global_free / 1024**3))
    print("occupied memory: %fGB" % (occupied / 1024**3))
    return global_free


def eval_main(cuda=True):
    args = parse_eval_args()
    print("args:", args)
    if args.wait_for_gpu_mem and get_cuda_free_memory() < 4000 * 1024 * 1024:
        print("free memory: %fGB" % (get_cuda_free_memory() / 1024**3))
        print("not enough memory, sleeping for 30s ~ 60s")
        time.sleep(30 + random.random() * 30)

    result_dir = Path(args.save_dir) / args.name
    if os.path.exists(result_dir):
        print("[WARNING] result dir <%s> already exists, will be overwritten" % result_dir)

    if not (args.ckpt is None):
        model = CorrectSegNet.load_from_checkpoint(args.ckpt)
    elif not (args.pl_dir is None):
        matched_files = glob.glob(os.path.join(args.pl_dir, "checkpoints", "*ckpt"))
        # sort based on epoch=xx-step=xx.ckpt
        matched_files = sorted(matched_files, key=lambda x: int(x.split("-")[0].split("=")[1]), reverse=True)
        if len(matched_files) == 0:
            raise ValueError("No checkpoint found in %s" % args.pl_dir)
        elif len(matched_files) > 1:
            print("More than one checkpoint found in %s" % args.pl_dir)
            print("Using the one with most trained: %s" % matched_files[0])
        ckpt_path = matched_files[0]
        model = CorrectSegNet.load_from_checkpoint(
            ckpt_path,
        )
    else:
        raise ValueError("Either ckpt or pl_dir must be specified.")
    os.makedirs(result_dir, exist_ok=True)

    if cuda:
        model = model.cuda()
    # Use pytorch lightning API to load the model including hparams
    model.eval()

    train_df = pd.read_csv(os.path.join(args.train_dir, "train_data.csv"))

    # TODO: refactor later regarding inappropriate file name train_data.csv
    test_df = pd.read_csv(os.path.join(args.test_dir, "train_data.csv"))
    if args.max_train_samples is not None:
        train_df = train_df.iloc[: args.max_train_samples]
    if args.max_test_samples is not None:
        test_df = test_df.iloc[: args.max_test_samples]
    train_dataset, val_dataset, test_dataset, whole_dataset = assemble_train_test_dataset(train_df, test_df, model)

    if args.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(10))
        val_dataset = torch.utils.data.Subset(val_dataset, range(10))
        test_dataset = torch.utils.data.Subset(test_dataset, range(10))

    # compute metrics
    print("[EVAL] computing metrics with threshold {}".format(args.out_threshold))
    train_metrics = compute_metrics(train_dataset, model, out_threshold=args.out_threshold, whole_dataset=whole_dataset)
    val_metrics = compute_metrics(val_dataset, model, out_threshold=args.out_threshold, whole_dataset=whole_dataset)
    test_metrics = compute_metrics(test_dataset, model, out_threshold=args.out_threshold, whole_dataset=None)

    # save metrics
    print("[EVAL] saving metrics")

    avg_train_metrics = {key: np.mean(value) for key, value in train_metrics.items()}
    avg_val_metrics = {key: np.mean(value) for key, value in val_metrics.items()}
    avg_test_metrics = {key: np.mean(value) for key, value in test_metrics.items()}

    metrics_df = pd.DataFrame(
        {
            "train": pd.Series(avg_train_metrics),
            "val": pd.Series(avg_val_metrics),
            "test": pd.Series(avg_test_metrics),
        }
    )

    metrics_df.to_csv(result_dir / "metrics.csv")
    print("[EVAL] metrics done")

    # visualize samples
    def _viz_samples(dataset, save_path):
        for i, sample in enumerate(tqdm.tqdm(dataset)):
            viz_sample_v3(
                sample, model, out_threshold=args.out_threshold, save_path=save_path / "sample-{}.png".format(i)
            )

    viz_fig_path = result_dir / "sample_viz"
    os.makedirs(viz_fig_path, exist_ok=True)
    os.makedirs(viz_fig_path / "train", exist_ok=True)
    os.makedirs(viz_fig_path / "val", exist_ok=True)
    os.makedirs(viz_fig_path / "test", exist_ok=True)

    if args.viz_pred:
        print("[EVAL] visualizing samples")
        _viz_samples(train_dataset, viz_fig_path / "train")
        _viz_samples(val_dataset, viz_fig_path / "val")
        _viz_samples(test_dataset, viz_fig_path / "test")
    else:
        print("[EVAL] skip visualizing samples...")

    print("[EVAL] done")


if __name__ == "__main__":
    eval_main()
