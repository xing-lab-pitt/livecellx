import argparse
import re

def get_avg_acc(log_file_path, epoch_num):
    with open(log_file_path, 'r') as f:
        logs = f.read()

    print(fr"Epoch\(train\)\s*\[{epoch_num}\]\[\s*\d+/\d+\].*top1_acc: (\d+\.\d+)")
    train_logs = re.findall(fr"Epoch\s*\(train\)  \[{epoch_num}\]\[\s*\d+/\d+\].*top1_acc: (\d+\.\d+)", logs)
    print("train_logs:", train_logs)
    train_top1_acc = [float(log) for log in train_logs]
    train_avg_acc = sum(train_top1_acc) / len(train_top1_acc)


    # TODO: train can be inaccurate due to the last batch size, need weights or discard the last batch in the future
    print(f"Average accuracy of epoch {epoch_num} (train): {train_avg_acc}")

    val_logs = re.findall(fr"Epoch\(val\)  \[{epoch_num}\]\[\d+/\d+\].*acc/top1: (\d+\.\d+)", logs)
    print("val_logs", val_logs)
    val_top1_acc = [float(log) for log in val_logs]
    val_avg_acc = sum(val_top1_acc) / len(val_top1_acc)
    print(f"Top1 accuracy of epoch {epoch_num} (val): {val_avg_acc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average accuracy for a given epoch from log file')
    parser.add_argument('log_file_path', type=str, help='path to the log file')
    parser.add_argument('epoch_num', type=int, help='epoch number for which to calculate average accuracy', default=-1)
    args = parser.parse_args()
    
    # print args
    print("log_file_path:", args.log_file_path)
    print("epoch_num:", args.epoch_num)

    get_avg_acc(args.log_file_path, args.epoch_num)