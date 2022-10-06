import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from livecell_tracker.core.single_cell import SingleCellTrajectory


def generate_single_trajectory_movie(
    sc_traj: SingleCellTrajectory,
    raw_imgs,
    save_path="./tmp.gif",
    min_length=10,
    ax=None,
    fig=None,
):
    if ax is not None and fig is None:
        fig = plt.gcf()
    elif ax is None and fig is not None:
        ax = plt.gca()
    elif ax is None and fig is None:
        fig, ax = plt.subplots()
    else:
        pass
    sc_traj.generate_single_trajectory_movie(min_length=min_length, fig=fig, ax=ax, save_path=save_path)
