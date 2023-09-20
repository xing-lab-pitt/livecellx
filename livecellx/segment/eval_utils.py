import seaborn as sns
from pathlib import Path
import json
import matplotlib.pyplot as plt


def read_detectron_metrics_data(metric_path):
    metric_data = []
    with open(metric_path, "r") as f:
        for line in f:
            metric_data.append(json.loads(line))
    return metric_data


def plot_train_curve(metric_path, ax=None, dpi=80, figsize=(16, 12), **kwargs):
    metric_data = read_detectron_metrics_data(metric_path)
    # print(metric_data[0].keys())
    total_loss = [x["total_loss"] for x in metric_data]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=80)
    ax.plot(total_loss, **kwargs)
    return ax
