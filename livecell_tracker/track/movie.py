import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from livecell_tracker.core.single_cell import SingleCellTrajectory


def generate_single_trajectory_movie(
    sc_traj: SingleCellTrajectory,
    raw_imgs,
    save_path="./tmp.gif",
    min_length=10,
):
    fig, ax = plt.subplots()
    sc_traj.generate_single_trajectory_movie(min_length=min_length, fig=fig, ax=ax, save_path=save_path)
