import argparse
import re
from pathlib import Path

import pandas as pd


def _get_avg_scc(logs, epoch_num):
    train_logs = re.findall(rf"Epoch\(train\).*\[{epoch_num}\]\[\s*\d+/\d+\].*top1_acc: (\d+\.\d+)", logs)
    # print("train_logs:", train_logs)
    if train_logs == []:
        return None, None
    train_top1_acc = [float(log) for log in train_logs]
    train_avg_acc = sum(train_top1_acc) / len(train_top1_acc)

    val_logs = re.findall(rf"Epoch\(val\).*\[{epoch_num}\]\[\d+/\d+\].*acc/top1: (\d+\.\d+)", logs)
    # print("val_logs", val_logs)
    val_top1_acc = [float(log) for log in val_logs]
    val_avg_acc = sum(val_top1_acc) / len(val_top1_acc)
    return train_avg_acc, val_avg_acc


def prune_log_filename(log_filename):
    start_str = "rgb-"
    end_str = "."

    start_index = log_filename.find(start_str) + len(start_str)
    end_index = log_filename.find(end_str)

    extracted_str = log_filename[start_index : end_index + 1]
    return extracted_str


def get_avg_acc(log_file_path, epoch_num):
    with open(log_file_path, "r") as f:
        logs = f.read()
    train_avg_acc, val_avg_acc = _get_avg_scc(logs, epoch_num)

    # TODO: train can be inaccurate due to the last batch size, need weights or discard the last batch in the future
    print(f"Average accuracy of epoch {epoch_num} (train): {train_avg_acc}")
    print(f"Top1 accuracy of epoch {epoch_num} (val): {val_avg_acc}")


def plot_avg_acc(log_file_path, epoch_num, start_epoch=1, out_dir: Path = Path("result_plots/")):
    with open(log_file_path, "r") as f:
        logs = f.read()

    # plot acc curve
    train_avg_accs = []
    val_avg_accs = []
    end_epoch = start_epoch + epoch_num
    for epoch in range(start_epoch, end_epoch + 1):
        train_avg_acc, val_avg_acc = _get_avg_scc(logs, epoch)
        train_avg_accs.append(train_avg_acc)
        val_avg_accs.append(val_avg_acc)
    # plot and save the fig
    import matplotlib.pyplot as plt

    plt.plot(range(start_epoch, end_epoch + 1), train_avg_accs, label="train")
    plt.plot(range(start_epoch, end_epoch + 1), val_avg_accs, label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    # get basename
    import os

    # get parent's parent dir name as the log filename
    # make log file path a posix path
    log_file_path = Path(os.path.abspath(log_file_path)).as_posix()
    log_filename = str(Path(log_file_path).parent.parent.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / str("acc-" + log_filename.replace(".log", ".png"))
    print("saving fig to:", fig_path)
    plt.savefig(fig_path)
    plt.close()

    # save the accs to a csv txt file
    df = pd.DataFrame(
        {"epoch": range(start_epoch, end_epoch + 1), "train_avg_acc": train_avg_accs, "val_avg_acc": val_avg_accs}
    )

    extracted_str = prune_log_filename(log_filename)
    df.to_csv(out_dir / str("acc-" + extracted_str + ".csv"), index=False)
    return df


def show_result_plots():
    paths = [
        # r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-drop-div-combined-clipLen=1-trainClipNum=1-valClipNum=1/20230824_003503/20230824_003503.log",
        # r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-combined-clipLen=1-trainClipNum=3-valClipNum=3/20230811_130716/20230811_130716.log",
        # r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-combined-clipLen=2-trainClipNum=3-valClipNum=3/20230717_030248/20230717_030248.log",
        # r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-combined-clipLen=3-trainClipNum=3-valClipNum=3/20230719_181939/20230719_181939.log",
        # r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-video-clipLen=1-trainClipNum=3-valClipNum=3/20230811_124019/20230811_124019.log",
        # r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-video-clipLen=2-trainClipNum=3-valClipNum=3/20230811_124027/20230811_124027.log",
        # r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-video-clipLen=3-trainClipNum=3-valClipNum=3/20230721_045839/20230721_045839.log",
        # r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v11-drop-div-combined-clipLen=3-trainClipNum=3-valClipNum=3/20230901_033504/20230901_033504.log",
        # r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v11-drop-div-video-clipLen=3-trainClipNum=3-valClipNum=3/20230901_033551/20230901_033551.log",
        # r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v11-st-combined-clipLen=3-trainClipNum=3-valClipNum=3/20230909_200322/20230909_200322.log",
        # r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v11-st-video-clipLen=3-trainClipNum=3-valClipNum=3/20230901_033807/20230901_033807.log",
        # r"./work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v12-st-combined-clipLen=1-trainClipNum=1-valClipNum=1/20231005_041430/20231005_041430.log",
        #  r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v12-st-combined-clipLen=2-trainClipNum=3-valClipNum=3/20230928_023147/20230928_023147.log",
        # r"./work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v12-st-combined-clipLen=3-trainClipNum=3-valClipNum=3/20231005_043230/20231005_043230.log",
        # r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v12-st-video-clipLen=2-trainClipNum=3-valClipNum=3/20231001_154756/20231001_154756.log"
        # r"./work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v13-drop-div-combined-clipLen=2-trainClipNum=3-valClipNum=3/20231019_003517/20231019_003517.log",
        # r"./work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v13-drop-div-combined-clipLen=3-trainClipNum=3-valClipNum=3/20231019_003517/20231019_003517.log",
        # r"./work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v13-drop-div-video-clipLen=2-trainClipNum=3-valClipNum=3/20231019_003517/20231019_003517.log",
        # r"./work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v13-drop-div-video-clipLen=3-trainClipNum=3-valClipNum=3/20231019_003517/20231019_003517.log",
        # r"./work_dirs/timesformer-default-divst-v13-st-video-random-crop/20231021_235643/20231021_235643.log",
        # r"./work_dirs/timesformer-default-divst-v13-st-combined-random-crop/20231021_134335/20231021_134335.log",
        # r"./work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v13-inclusive-corrected-v1-all-clipLen=3-trainClipNum=3-valClipNum=3/20231029_141110/20231029_141110.log",
        r"./work_dirs/timesformer-default-divst-v14-inclusive-combined-random-crop/20240130_063549/20240130_063549.log"
    ]
    epoch = 20
    all_df = None
    for path in paths:
        print(">>> path:", path)
        get_avg_acc(path, epoch)
        res_df = plot_avg_acc(path, epoch)
        res_df["model"] = prune_log_filename(path)
        if all_df is not None:
            all_df = pd.concat([all_df, plot_avg_acc(path, epoch)])
        else:
            all_df = plot_avg_acc(path, epoch)
        print("-" * 50)
    all_df.to_csv("result_plots/acc-compare.csv", index=False)


if __name__ == "__main__":
    # show_result_plots()
    # exit(0)

    parser = argparse.ArgumentParser(description="Calculate average accuracy for a given epoch from log file")
    parser.add_argument("log_file_path", type=str, help="path to the log file")
    parser.add_argument("epoch_num", type=int, help="epoch number for which to calculate average accuracy", default=-1)
    parser.add_argument("--start_epoch", type=int, help="start epoch number for plotting", default=1)
    args = parser.parse_args()

    # print args
    print("log_file_path:", args.log_file_path)
    print("epoch_num:", args.epoch_num)

    get_avg_acc(args.log_file_path, args.start_epoch + args.epoch_num)
    plot_avg_acc(args.log_file_path, args.epoch_num, start_epoch=args.start_epoch)
