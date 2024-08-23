from typing import Callable, List
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
import numpy as np
from pathlib import Path

from livecellx.core.single_cell import SingleCellStatic, SingleCellTrajectory, get_time2scs
from livecellx.core.io_utils import save_png
from livecellx.core.datasets import LiveCellImageDataset
from livecellx.preprocess.utils import normalize_img_to_uint8


def generate_scs_movie(
    scs: List[SingleCellStatic],
    img_dataset: LiveCellImageDataset,
    save_dir,
    fps=3,
    factor=0.5,
    video_only=False,
    use_all_imgs=True,
):
    """
    Generate a movie from a list of SingleCellStatic objects and an image dataset.
    Note: pip install imageio[ffmpeg] imageio[pyav] is required, read https://github.com/OpenTalker/SadTalker/issues/276

    Args:
        scs (List[SingleCellStatic]): List of SingleCellStatic objects representing the cells to be tracked.
        img_dataset (LiveCellImageDataset): Image dataset containing the images corresponding to the cells.
        save_dir (str): Directory to save the generated movie.
        fps (int, optional): Frames per second of the generated movie. Defaults to 3.
        factor (float, optional): Factor to adjust the intensity of the cells in the movie. Defaults to 0.5.
        video_only (bool, optional): Flag indicating whether to save only the video file without individual frames. Defaults to False.
        use_all_imgs (bool, optional): Flag indicating whether to use all images in the dataset or only the ones corresponding to the tracked cells. Defaults to True.
    """

    time2scs = get_time2scs(scs)
    if use_all_imgs:
        times = img_dataset.times
    else:
        times = sorted(time2scs.keys())
    save_dir = Path(save_dir)
    img_dir = save_dir / "imgs"
    # Create
    img_dir.mkdir(exist_ok=True, parents=True)
    # Create a movie writer object
    with imageio.get_writer(save_dir / "movie.mp4", fps=fps) as writer:
        for time in times:
            if time in time2scs:
                cur_scs = time2scs[time]
            else:
                cur_scs = []
            img = img_dataset.get_img_by_time(time)
            img = np.array(img)
            img = normalize_img_to_uint8(img)
            # RGBA
            img = np.repeat(img[:, :, np.newaxis], 4, axis=2)
            img[..., 0] = 0
            img[..., 3] = 255
            for sc in cur_scs:
                sc_mask = sc.get_sc_mask(crop=False)
                assert img.shape[:2] == sc_mask.shape[:2], "img and mask shape mismatch"
                for sc in cur_scs:
                    img[..., 0][sc_mask] += int(255 * factor)
            # Clip the value to 255
            img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)
            # Save image as png
            # Mode should be corresponding RGBA code in PIL
            save_png(img, img_dir / (str(time) + ".png"), mode="RGBA")
            # Append the image to the movie file
            writer.append_data(img)


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
            print(
                "[Viz] skipping the current trajectory track_id: ",
                single_cell_trajectory.track_id,
            )
            return None
    if ax is None:
        fig, ax = plt.subplots()

    def init():
        return []

    def default_update(sc_tp: SingleCellStatic, draw_contour=True):
        frame_idx, raw_img, bbox, img_crop = (
            sc_tp.timeframe,
            sc_tp.get_img(),
            sc_tp.bbox,
            sc_tp.get_img_crop(),
        )
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
            contour_coords = sc_tp.get_contour_coords_on_img_crop()
            ax.scatter(contour_coords[:, 1], contour_coords[:, 0], s=2, c="r")
        return []

    if ani_update_func is None:
        ani_update_func = default_update

    frame_data = []
    for frame_idx in single_cell_trajectory.timeframe_to_single_cell:
        sc_timepoint = single_cell_trajectory.get_single_cell(frame_idx)
        frame_data.append(sc_timepoint)

    ani = FuncAnimation(fig, default_update, frames=frame_data, init_func=init, blit=True)
    print("saving to: %s..." % save_path)
    ani.save(save_path)
