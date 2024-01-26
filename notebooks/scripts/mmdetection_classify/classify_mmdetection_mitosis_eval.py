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

from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score

# from livecellx import segment
from livecellx import core
from livecellx.core import datasets
from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset
from skimage import measure
from livecellx.core import SingleCellTrajectory, SingleCellStatic
from livecellx.core.sc_video_utils import gen_mp4_from_frames, combine_video_frames_and_masks
from livecellx.preprocess.utils import normalize_img_to_uint8
from livecellx.core.io_utils import save_png

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
import livecellx.track.timesformer_inference
import mmcv
from mmaction.apis import init_recognizer, inference_recognizer

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %%
import livecellx.track.timesformer_inference


import argparse

parser = argparse.ArgumentParser(description="Evaluate mitosis detection using mmdetection")
parser.add_argument("--model_dir", type=str, help="Path to the directory containing the model", required=True)
parser.add_argument("--out_dir", type=str, help="Path to the output directory", required=True)
parser.add_argument("--config", type=str, help="Path to the configuration file", required=False, default=None)
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
    help="[Deprecated] Path to the mmaction data tsv file",
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
parser.add_argument("--is-tsn", action="store_true", help="if it is tsn model")
parser.add_argument(
    "--raw-video-treat-as-negative",
    action="store_true",
    help="In combined ver, raw videos input labels should be all WRONG",
    default=False,
)

parser.add_argument(
    "--no-wrong-video",
    action="store_true",
    help="whether to store wrong videos",
    default=False,
)


args = parser.parse_args()

out_wrong_video_dir = Path(args.out_dir) / "wrong_videos"
out_wrong_video_dir.mkdir(parents=True, exist_ok=True)

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
if config_file is None:
    # a python file starting with 'config' and ends with '.py'
    _files = glob.glob(os.path.join(model_dir, "config*.py"))
    if len(_files) == 0:
        raise ValueError("config file not found")
    elif len(_files) > 1:
        print("WARNING: multiple config files found, using the first one: ", _files)
    config_file = _files[0]
    print("config_file:", config_file)
    assert config_file is not None
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

from tqdm import tqdm

print("test data frame:", test_data_df.columns[:2])

if args.add_random_crop and args.is_tsn:
    model.cfg.test_pipeline = [
        dict(io_backend="disk", type="DecordInit"),
        dict(clip_len=3, frame_interval=1, num_clips=3, test_mode=True, type="SampleFrames"),
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
        dict(
            scale=(
                -1,
                256,
            ),
            type="Resize",
        ),
        dict(crop_size=224, type="TenCrop"),
        dict(input_format="NCHW", type="FormatShape"),
        dict(type="PackActionInputs"),
    ]
elif args.add_random_crop:
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
    input_frame_type = row_series["frame_type"]
    padding_pixels = row_series["padding_pixels"]
    try:
        model.zero_grad()
        results, data, grad = livecellx.track.timesformer_inference.inference_recognizer(
            model, video_path, require_grad=True, return_data_and_grad=True
        )
    except Exception as e:
        print("exception during prediction:", e)
        # print backtrace
        import traceback

        traceback.print_exc()
        print("video_path:", video_path)
        continue
    # predicted_label = results.pred_labels.item.cpu().numpy()[0]

    predicted_label = results.pred_label.item()
    if args.raw_video_treat_as_negative:
        test_gt_label = 2  # no focus!
    else:
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

        # Do not save wrong videos if specified
        if args.no_wrong_video:
            continue

        data_input = data["inputs"][0]  # 3 x 3 x 8 x 224 x 224
        if not args.is_tsn:
            # timeSformer
            imgs = data_input[1][2].detach().cpu().numpy()  # 8 x 224 x 224
            masks = data_input[1][0].detach().cpu().numpy()  # 8 x 224 x 224
            imgs = list(imgs)
            masks = list(masks)
            imgs = [normalize_img_to_uint8(img) for img in imgs]
            masks = [normalize_img_to_uint8(mask) for mask in masks]
            # Extract video filename from video_path without extension
            video_filename = Path(video_path).name
            _save_path = out_wrong_video_dir / video_filename
            # already edt transformed, so set to false
            frames = combine_video_frames_and_masks(imgs, masks, is_gray=True, edt_transform=False)
            gen_mp4_from_frames(frames, _save_path, fps=3)
        else:
            # tsn
            imgs = data_input.detach().cpu().numpy()  # 90 x 3 x h x w
            imgs = list(imgs)
            imgs = [normalize_img_to_uint8(img) for img in imgs]

            # extract video filename from video_path without extension
            video_filename = Path(video_path).stem
            tmp_out_sample_dir = out_wrong_video_dir / video_filename
            tmp_out_sample_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(imgs):
                save_png(tmp_out_sample_dir / f"sample_dim0-{i}.png", img.swapaxes(0, 2), mode="RGB")
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


def report_classification_metrics(true_labels, predicted_labels):
    # calculate the accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # calculate the precision
    precision = precision_score(true_labels, predicted_labels, average="weighted")

    # calculate the F1 score
    f1 = f1_score(true_labels, predicted_labels, average="weighted")

    # generate a classification report
    report = classification_report(true_labels, predicted_labels, digits=4)

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
frame_types = all_predictions_df["frame_type"].unique()

for frame_type in frame_types:
    indexer = all_predictions_df["frame_type"] == frame_type
    print("#" * 40, "frame_type:", frame_type, "#" * 40)
    report_classification_metrics(
        all_predictions_df[indexer]["true_label"], all_predictions_df[indexer]["predicted_label"]
    )

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
