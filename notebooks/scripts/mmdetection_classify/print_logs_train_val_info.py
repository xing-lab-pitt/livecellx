import argparse
import re

def get_avg_acc(log_file_path, epoch_num):
    with open(log_file_path, 'r') as f:
        logs = f.read()

    
    train_logs = re.findall(fr"Epoch\(train\).*\[{epoch_num}\]\[\s*\d+/\d+\].*top1_acc: (\d+\.\d+)", logs)
    print("train_logs:", train_logs)
    train_top1_acc = [float(log) for log in train_logs]
    train_avg_acc = sum(train_top1_acc) / len(train_top1_acc)


    val_logs = re.findall(fr"Epoch\(val\).*\[{epoch_num}\]\[\d+/\d+\].*acc/top1: (\d+\.\d+)", logs)
    print("val_logs", val_logs)
    val_top1_acc = [float(log) for log in val_logs]
    val_avg_acc = sum(val_top1_acc) / len(val_top1_acc)


    # TODO: train can be inaccurate due to the last batch size, need weights or discard the last batch in the future
    print(f"Average accuracy of epoch {epoch_num} (train): {train_avg_acc}")
    print(f"Top1 accuracy of epoch {epoch_num} (val): {val_avg_acc}")



def show_results_v10_st():
    paths = [r"/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-combined-clipLen=1-trainClipNum=3-valClipNum=3/20230811_130716/20230811_130716.log",
             r"/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-combined-clipLen=2-trainClipNum=3-valClipNum=3/20230717_030248/20230717_030248.log",
             r"/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-combined-clipLen=3-trainClipNum=3-valClipNum=3/20230719_181939/20230719_181939.log",
             r"/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-video-clipLen=1-trainClipNum=3-valClipNum=3/20230811_124019/20230811_124019.log",
             r"/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-video-clipLen=2-trainClipNum=3-valClipNum=3/20230811_124027/20230811_124027.log",
             r"/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v10-st-video-clipLen=3-trainClipNum=3-valClipNum=3/20230721_045839/20230721_045839.log"
             ]
    epoch = 200
    for path in paths:
        print(">>> path:", path)
        get_avg_acc(path, epoch)
        print("-" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average accuracy for a given epoch from log file')
    parser.add_argument('log_file_path', type=str, help='path to the log file')
    parser.add_argument('epoch_num', type=int, help='epoch number for which to calculate average accuracy', default=-1)
    args = parser.parse_args()
    
    # print args
    print("log_file_path:", args.log_file_path)
    print("epoch_num:", args.epoch_num)

    get_avg_acc(args.log_file_path, args.epoch_num)
    # show_results_v10_st()