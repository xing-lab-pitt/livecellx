import seaborn as sns
from pathlib import Path
import json
import matplotlib.pyplot as plt

from livecell_tracker.segment.eval_utils import read_detectron_metrics_data, plot_train_curve

# TODO
def plot_metrics_results():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=80)
    metric_path = ...
    plot_train_curve(metric_path, label="3-expert", ax=ax)
