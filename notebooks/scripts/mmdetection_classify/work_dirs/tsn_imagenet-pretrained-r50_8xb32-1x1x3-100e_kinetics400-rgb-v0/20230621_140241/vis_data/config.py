model = dict(
    type="Recognizer2D",
    backbone=dict(
        type="ResNet", pretrained="https://download.pytorch.org/models/resnet50-11ad3fa6.pth", depth=50, norm_eval=False
    ),
    cls_head=dict(
        type="TSNHead",
        num_classes=2,
        in_channels=2048,
        spatial_type="avg",
        consensus=dict(type="AvgConsensus", dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips="prob",
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], format_shape="NCHW"
    ),
    train_cfg=None,
    test_cfg=None,
)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
param_scheduler = [dict(type="MultiStepLR", begin=0, end=100, by_epoch=True, milestones=[40, 80], gamma=0.1)]
optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001), clip_grad=dict(max_norm=40, norm_type=2)
)
default_scope = "mmaction"
default_hooks = dict(
    runtime_info=dict(type="RuntimeInfoHook"),
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20, ignore_last=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=3, save_best="auto", max_keep_ckpts=3),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffers=dict(type="SyncBuffersHook"),
)
env_cfg = dict(
    cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0), dist_cfg=dict(backend="nccl")
)
log_processor = dict(type="LogProcessor", window_size=20, by_epoch=True)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="ActionVisualizer", vis_backends=[dict(type="LocalVisBackend")])
log_level = "INFO"
load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth"
resume = False
dataset_type = "VideoDataset"
data_root = "../../notebook_results/mmaction_train_data/all_data.csv"
data_root_val = "../../notebook_results/mmaction_train_data/all_data.csv"
ann_file_train = "../../notebook_results/mmaction_train_data/train_data.csv"
ann_file_val = "../../notebook_results/mmaction_train_data/test_data.csv"
file_client_args = dict(io_backend="disk")
train_pipeline = [
    dict(type="DecordInit", io_backend="disk"),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=3),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="MultiScaleCrop", input_size=224, scales=(1, 0.875, 0.75, 0.66), random_crop=False, max_wh_scale_gap=1),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="DecordInit", io_backend="disk"),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=3, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]
test_pipeline = [
    dict(type="DecordInit", io_backend="disk"),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=25, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="TenCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="VideoDataset",
        ann_file="../../notebook_results/mmaction_train_data/train_data.csv",
        data_prefix=dict(video="../../notebook_results/mmaction_train_data/all_data.csv"),
        pipeline=[
            dict(type="DecordInit", io_backend="disk"),
            dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=3),
            dict(type="DecordDecode"),
            dict(type="Resize", scale=(-1, 256)),
            dict(
                type="MultiScaleCrop",
                input_size=224,
                scales=(1, 0.875, 0.75, 0.66),
                random_crop=False,
                max_wh_scale_gap=1,
            ),
            dict(type="Resize", scale=(224, 224), keep_ratio=False),
            dict(type="Flip", flip_ratio=0.5),
            dict(type="FormatShape", input_format="NCHW"),
            dict(type="PackActionInputs"),
        ],
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="VideoDataset",
        ann_file="../../notebook_results/mmaction_train_data/test_data.csv",
        data_prefix=dict(video="../../notebook_results/mmaction_train_data/all_data.csv"),
        pipeline=[
            dict(type="DecordInit", io_backend="disk"),
            dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=3, test_mode=True),
            dict(type="DecordDecode"),
            dict(type="Resize", scale=(-1, 256)),
            dict(type="CenterCrop", crop_size=224),
            dict(type="FormatShape", input_format="NCHW"),
            dict(type="PackActionInputs"),
        ],
        test_mode=True,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="VideoDataset",
        ann_file="../../notebook_results/mmaction_train_data/test_data.csv",
        data_prefix=dict(video="../../notebook_results/mmaction_train_data/all_data.csv"),
        pipeline=[
            dict(type="DecordInit", io_backend="disk"),
            dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=25, test_mode=True),
            dict(type="DecordDecode"),
            dict(type="Resize", scale=(-1, 256)),
            dict(type="TenCrop", crop_size=224),
            dict(type="FormatShape", input_format="NCHW"),
            dict(type="PackActionInputs"),
        ],
        test_mode=True,
    ),
)
val_evaluator = dict(type="AccMetric")
test_evaluator = dict(type="AccMetric")
auto_scale_lr = dict(enable=False, base_batch_size=256)
launcher = "none"
work_dir = "./work_dirs\\tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb"
randomness = dict(seed=None, diff_rank_seed=False, deterministic=False)
