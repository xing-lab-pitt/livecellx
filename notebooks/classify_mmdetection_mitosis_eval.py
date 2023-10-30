# %%
import numpy as np
import matplotlib.pyplot as plt

# from cellpose import models
from cellpose.io import imread
import glob
from pathlib import Path
from PIL import Image, ImageSequence
from tqdm import tqdm
import os
import os.path

# from livecellx import segment
from livecellx import core
from livecellx.core import datasets
from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset
from skimage import measure
from livecellx.core import SingleCellTrajectory, SingleCellStatic

# import detectron2
# from detectron2.utils.logger import setup_logger

# setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import cv2

# %% [markdown]
# ## Load model

# %%
import torch
from pathlib import Path
import pandas as pd

# %%
from mmengine.config import Config, DictAction
from mmaction.registry import MODELS
import livecellx.track.customized_inference
import mmcv
from mmaction.apis import init_recognizer, inference_recognizer

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %%
import livecellx.track.customized_inference


import argparse

parser = argparse.ArgumentParser(description="Evaluate mitosis detection using mmdetection")
parser.add_argument("--model_dir", type=str, help="Path to the directory containing the model", required=True)
parser.add_argument("--out_dir", type=str, help="Path to the output directory", required=True)
parser.add_argument("--config", type=str, help="Path to the configuration file", required=True)
parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file", required=True)
parser.add_argument(
    "--video_dir",
    type=str,
    help="Path to the video directory",
    default=r"./notebook_results/mmaction_train_data_v13-inclusive-corrected/videos",
)
parser.add_argument(
    "--mmaction_data_tsv",
    type=str,
    help="Path to the mmaction data tsv file",
    default=r"./notebook_results/mmaction_train_data_v13-inclusive-corrected/mmaction_test_data_all.txt",
)
parser.add_argument(
    "--test_data_meta_path",
    type=str,
    help="Path to the test data meta file",
    default=r"./notebook_results/mmaction_train_data_v13-inclusive-corrected/test_data.txt",
)
parser.add_argument("--device", type=str, help="Device to use", default="cuda:0")
parser.add_argument("--add-random-crop", action="store_true", help="Add random crop to the pipeline")


args = parser.parse_args()

print(
    "#" * 40,
    "args",
    "#" * 40,
)
# print each arg on each line
for arg in vars(args):
    print(arg, ":", getattr(args, arg))

model_dir = args.model_dir
out_dir = Path(args.out_dir)
print("creating out_dir:", out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

checkpoint_file = args.checkpoint
config_file = args.config
cfg = Config.fromfile(config_file)
DEVICE = args.device
model = init_recognizer(config_file, checkpoint_file, device=DEVICE)
test_data_meta_path = args.test_data_meta_path
mmaction_data_tsv = args.mmaction_data_tsv
video_dir = args.video_dir

# %%
# test dataframe
test_data_df = pd.read_csv(test_data_meta_path, sep=" ")
test_data_df = test_data_df.rename(columns={"label_index": "label"})
print(test_data_df.shape)
test_data_df[:2]

# mmaction df columns are path and label
mmaction_data_df = pd.read_csv(mmaction_data_tsv, sep=" ", header=None)
mmaction_data_df.columns = ["path", "label"]
print(mmaction_data_df.shape)
mmaction_data_df[:2]
from tqdm import tqdm

print("test data frame:", test_data_df.columns[:2])

if args.add_random_crop:
    model.cfg.test_pipeline = [
        dict(io_backend="disk", type="DecordInit"),
        dict(clip_len=8, frame_interval=1, num_clips=1, test_mode=True, type="SampleFrames"),
        dict(type="DecordDecode"),
        dict(
            scale_range=(
                224,
                300,
            ),
            type="RandomRescale",
        ),
        dict(size=224, type="RandomCrop"),
        dict(
            scale=(
                -1,
                224,
            ),
            type="Resize",
        ),
        dict(crop_size=224, type="ThreeCrop"),
        dict(input_format="NCTHW", type="FormatShape"),
        dict(type="PackActionInputs"),
    ]
else:
    pass
    # model.cfg.test_pipeline = [
    #             dict(io_backend='disk', type='DecordInit'),
    #             dict(
    #                 clip_len=8,
    #                 frame_interval=1,
    #                 num_clips=1,
    #                 test_mode=True,
    #                 type='SampleFrames'),
    #             dict(type='DecordDecode'),
    #             # dict(scale_range=(
    #             #     224,
    #             #     300,
    #             # ), type='RandomRescale'),
    #             # dict(scale_range=(
    #             #     224,
    #             #     300,
    #             # ), type='RandomRescale'),
    #             # dict(size=224, type='RandomCrop'),
    #             dict(scale=(
    #                 -1,
    #                 224,
    #             ), type='Resize'),
    #             dict(crop_size=224, type='ThreeCrop'),
    #             dict(input_format='NCTHW', type='FormatShape'),
    #             dict(type='PackActionInputs'),
    #         ]


nclasses = 3
gt2total = {class_idx: 0 for class_idx in range(nclasses)}
gt2correct = {class_idx: 0 for class_idx in range(nclasses)}
wrong_predictions = []
all_predictions = []
video_dir = Path(video_dir)
print("total #rows:", len(test_data_df))

all_rows = [_row for _row in test_data_df.iterrows()]
for row_ in tqdm(all_rows):
    idx, row_series = row_
    video_path = str(video_dir / row_series["path"])
    try:
        model.zero_grad()
        results, data, grad = livecellx.track.customized_inference.inference_recognizer(
            model, video_path, require_grad=True, return_data_and_grad=True
        )
    except Exception as e:
        print("exception during prediction:", e)
        # print backtrace
        import traceback

        traceback.print_exc()
        print("video_path:", video_path)
    # predicted_label = results.pred_labels.item.cpu().numpy()[0]
    predicted_label = results.pred_label.item()
    test_gt_label = row_series["label"]
    if test_gt_label not in gt2total:
        gt2total[test_gt_label] = 0
        gt2correct[test_gt_label] = 0
    gt2total[test_gt_label] += 1
    row_series = row_series.copy()
    row_series["predicted_label"] = predicted_label
    row_series["true_label"] = test_gt_label
    row_series["correct"] = predicted_label == test_gt_label
    all_predictions.append(row_series)
    if predicted_label != test_gt_label:
        print("wrong prediction:", video_path, "predicted_label:", predicted_label, "gt_label:", test_gt_label)
        wrong_predictions.append(row_series)
    else:
        gt2correct[test_gt_label] += 1


# %%
for test_gt_label, total in gt2total.items():
    correct = gt2correct[test_gt_label]
    if total == 0:
        print("no video for label:", test_gt_label)
        continue
    print("gt_label:", test_gt_label, "total:", total, "correct:", correct, "acc:", correct / total)

# %%
# convert all series in wrong_predictions to dataframe
all_predictions_df = pd.DataFrame(all_predictions)
wrong_predictions_df = pd.DataFrame(wrong_predictions)
wrong_predictions_df[:2]

# %%
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score


def report_classification_metrics(true_labels, predicted_labels):
    # calculate the accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # calculate the precision
    precision = precision_score(true_labels, predicted_labels, average="weighted")

    # calculate the F1 score
    f1 = f1_score(true_labels, predicted_labels, average="weighted")

    # generate a classification report
    report = classification_report(true_labels, predicted_labels)

    # print the metrics and classification report
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, precision, f1, report


# %%
report_classification_metrics(
    all_predictions_df[all_predictions_df["frame_type"] == "combined"]["true_label"],
    all_predictions_df[all_predictions_df["frame_type"] == "combined"]["predicted_label"],
)

# %%
indexer = all_predictions_df["frame_type"] == "combined"
report_classification_metrics(all_predictions_df[indexer]["true_label"], all_predictions_df[indexer]["predicted_label"])

# %%
all_predictions_df.to_csv(out_dir / "all_predictions.csv", index=False)
wrong_predictions_df.to_csv(out_dir / "wrong_predictions.csv", index=False)
wrong_predictions_df = pd.read_csv(out_dir / "wrong_predictions.csv")


# %% [markdown]
# ### Visualize wrong predictions according to three frame types: combined, raw and mask

# %%
# Extract the last three levels of the src_dir path
wrong_predictions_df["short_src_dir"] = (
    wrong_predictions_df["src_dir"].str.split(r"\\|/").apply(lambda x: "/".join(x[-3:]))
)
