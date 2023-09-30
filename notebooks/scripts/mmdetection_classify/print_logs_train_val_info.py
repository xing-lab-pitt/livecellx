import argparse
import re
from pathlib import Path


def _get_avg_scc(logs, epoch_num):
    train_logs = re.findall(rf"Epoch\(train\).*\[{epoch_num}\]\[\s*\d+/\d+\].*top1_acc: (\d+\.\d+)", logs)
    # print("train_logs:", train_logs)
    train_top1_acc = [float(log) for log in train_logs]
    train_avg_acc = sum(train_top1_acc) / len(train_top1_acc)

    val_logs = re.findall(rf"Epoch\(val\).*\[{epoch_num}\]\[\d+/\d+\].*acc/top1: (\d+\.\d+)", logs)
    # print("val_logs", val_logs)
    val_top1_acc = [float(log) for log in val_logs]
    val_avg_acc = sum(val_top1_acc) / len(val_top1_acc)
    return train_avg_acc, val_avg_acc


def get_avg_acc(log_file_path, epoch_num):
    with open(log_file_path, "r") as f:
        logs = f.read()
    train_avg_acc, val_avg_acc = _get_avg_scc(logs, epoch_num)

    # TODO: train can be inaccurate due to the last batch size, need weights or discard the last batch in the future
    print(f"Average accuracy of epoch {epoch_num} (train): {train_avg_acc}")
    print(f"Top1 accuracy of epoch {epoch_num} (val): {val_avg_acc}")


def plot_avg_acc(log_file_path, epoch_num, out_dir: Path = Path("result_plots/")):
    with open(log_file_path, "r") as f:
        logs = f.read()

    # plot acc curve
    train_avg_accs = []
    val_avg_accs = []
    for epoch in range(1, epoch_num + 1):
        train_avg_acc, val_avg_acc = _get_avg_scc(logs, epoch)
        train_avg_accs.append(train_avg_acc)
        val_avg_accs.append(val_avg_acc)
    # plot and save the fig
    import matplotlib.pyplot as plt

    plt.plot(range(1, epoch_num + 1), train_avg_accs, label="train")
    plt.plot(range(1, epoch_num + 1), val_avg_accs, label="test")
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
    plt.savefig(out_dir / str("acc-" + log_filename.replace(".log", ".png")))
    plt.close()


def show_results_v10_st():
    paths = [
        r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-combined-clipLen=1-trainClipNum=3-valClipNum=3/20230811_130716/20230811_130716.log",
        r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-combined-clipLen=2-trainClipNum=3-valClipNum=3/20230717_030248/20230717_030248.log",
        r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-combined-clipLen=3-trainClipNum=3-valClipNum=3/20230719_181939/20230719_181939.log",
        r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-video-clipLen=1-trainClipNum=3-valClipNum=3/20230811_124019/20230811_124019.log",
        r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-video-clipLen=2-trainClipNum=3-valClipNum=3/20230811_124027/20230811_124027.log",
        r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-video-clipLen=3-trainClipNum=3-valClipNum=3/20230721_045839/20230721_045839.log",
        r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v11-drop-div-combined-clipLen=3-trainClipNum=3-valClipNum=3/20230901_033504/20230901_033504.log",
        r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v11-drop-div-video-clipLen=3-trainClipNum=3-valClipNum=3/20230901_033551/20230901_033551.log",
        r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v11-st-combined-clipLen=3-trainClipNum=3-valClipNum=3/20230909_200322/20230909_200322.log",
        r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v11-st-video-clipLen=3-trainClipNum=3-valClipNum=3/20230901_033807/20230901_033807.log",
        #  r"/home/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v12-st-combined-clipLen=2-trainClipNum=3-valClipNum=3/20230928_023147/20230928_023147.log"
    ]
    epoch = 200
    for path in paths:
        print(">>> path:", path)
        get_avg_acc(path, epoch)
        plot_avg_acc(path, epoch)
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average accuracy for a given epoch from log file")
    parser.add_argument("log_file_path", type=str, help="path to the log file")
    parser.add_argument("epoch_num", type=int, help="epoch number for which to calculate average accuracy", default=-1)
    args = parser.parse_args()

    # print args
    print("log_file_path:", args.log_file_path)
    print("epoch_num:", args.epoch_num)

    get_avg_acc(args.log_file_path, args.epoch_num)
    plot_avg_acc(args.log_file_path, args.epoch_num)
    # show_results_v10_st()

    show_results_v10_st()
