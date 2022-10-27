from typing import Callable
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from livecell_tracker.core.single_cell import SingleCellStatic, SingleCellTrajectory


def generate_single_trajectory_movie(
    sc_traj: SingleCellTrajectory,
    raw_imgs,
    save_path="./tmp.mp4",
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


# TODO: refactor
def generate_single_trajectory_movie(
    single_cell_trajectory,
    save_path="./tmp.mp4",
    min_length=None,
    ax=None,
    fig=None,
    ani_update_func: Callable = None,  # how you draw each frame
):
    """generate movies of this single trajectory

    Parameters
    ----------
    save_path : str, optional
        _description_, by default "./tmp.gif"
    min_length : _type_, optional
        _description_, by default None
    ax : _type_, optional
        _description_, by default None
    fig : _type_, optional
        _description_, by default None
    ani_update_func : Callable, optional
        a callable function whose argument is a SingleCell object, by default None. This argument allows users to pass in customized functions to draw/beautify code.

    Returns
    -------
    _type_
        _description_
    """
    if min_length is not None:
        if single_cell_trajectory.get_timeframe_span_length() < min_length:
            print("[Viz] skipping the current trajectory track_id: ", single_cell_trajectory.track_id)
            return None
    if ax is None:
        fig, ax = plt.subplots()

    def init():
        return []

    def default_update(sc_tp: SingleCellStatic, draw_contour=True):
        frame_idx, raw_img, bbox, img_crop = (sc_tp.timeframe, sc_tp.get_img(), sc_tp.bbox, sc_tp.get_img_crop())
        ax.cla()
        frame_text = ax.text(
            -10,
            -10,
            "frame: {}".format(frame_idx),
            fontsize=10,
            color="red",
            ha="center",
            va="center",
        )
        ax.imshow(img_crop)

        if draw_contour:
            contour_coords = sc_tp.get_img_crop_contour_coords()
            ax.scatter(contour_coords[:, 1], contour_coords[:, 0], s=2, c="r")
        return []

    if ani_update_func is None:
        ani_update_func = default_update

    frame_data = []
    for frame_idx in single_cell_trajectory.timeframe_to_single_cell:
        sc_timepoint = single_cell_trajectory.get_single_cell(frame_idx)
        img = single_cell_trajectory.raw_img_dataset[frame_idx]
        bbox = sc_timepoint.get_bbox()
        frame_data.append(sc_timepoint)

    ani = FuncAnimation(fig, default_update, frames=frame_data, init_func=init, blit=True)
    print("saving to: %s..." % save_path)
    ani.save(save_path)
