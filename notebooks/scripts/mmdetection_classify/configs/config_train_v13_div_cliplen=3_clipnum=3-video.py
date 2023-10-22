# _base_ = [
#     '../../_base_/models/tsn_r50.py', '../../_base_/schedules/sgd_100e.py',
#     '../../_base_/default_runtime.py'
# ]


_base_ = [
    "./tsn_r50.py",
    "./sgd_500e.py",
    "./default_runtime.py",
]


# pretrained
model = dict(cls_head=dict(num_classes=3))
load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth"


# dataset settings
# dataset_type = 'VideoDataset'
# data_root = 'data/kinetics400/videos_train'
# data_root_val = 'data/kinetics400/videos_val'
# ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
# ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'

ver = "13-drop-div"
frame_type = "video"
# frame_type = "video"
CLIP_LEN = 3
TRAIN_CLIP_NUM = 3
VAL_CLIP_NUM = 3

data_dir = "../../notebook_results/mmaction_train_data_v" + str(ver) + "/"
dataset_type = "VideoDataset"
data_root = data_dir + "videos"
data_root_val = data_dir + "videos"

# for v0-v7
# ann_file_train = data_dir + "train_data_" + frame_type + ".txt"
# ann_file_val = data_dir + "test_data_" + frame_type + ".txt"

# for v8 onward
ann_file_train = data_dir + "mmaction_train_data_" + frame_type + ".txt"
ann_file_val = data_dir + "mmaction_test_data_" + frame_type + ".txt"

file_client_args = dict(io_backend="disk")
work_dir = f"./work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v{ver}-{frame_type}-clipLen={CLIP_LEN}-trainClipNum={TRAIN_CLIP_NUM}-valClipNum={VAL_CLIP_NUM}"
#########################################################
import cv2
import numpy as np
import math

##########################################################

train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    # dict(type="RawFrameDecode", channel_order="rgb", **file_client_args),
    dict(type="SampleFrames", clip_len=CLIP_LEN, frame_interval=1, num_clips=TRAIN_CLIP_NUM),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="MultiScaleCrop", input_size=224, scales=(1, 0.875, 0.75, 0.66), random_crop=False, max_wh_scale_gap=1),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="DecordInit", **file_client_args),
    # dict(type="RawFrameDecode", channel_order="rgb", **file_client_args),
    dict(type="SampleFrames", clip_len=CLIP_LEN, frame_interval=1, num_clips=VAL_CLIP_NUM, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]
test_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFrames", clip_len=CLIP_LEN, frame_interval=1, num_clips=VAL_CLIP_NUM, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="TenCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type, ann_file=ann_file_train, data_prefix=dict(video=data_root), pipeline=train_pipeline
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(type="AccMetric")
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=50, max_keep_ckpts=10))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=256)
