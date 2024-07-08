from typing import List, Tuple, Union
import numpy as np
from livecellx.core.single_cell import (
    SingleCellStatic,
    SingleCellTrajectory,
    SingleCellTrajectoryCollection,
    get_time2scs,
)
from livecellx.livecell_logger import main_warning


def split_mitosis_sample(
    sample: List[SingleCellStatic], return_break_idx=True
) -> Union[List[List[SingleCellStatic]], Tuple[List[List[SingleCellStatic]], int]]:
    """
    Splits a list of single-cell static objects into multiple trajectories based on mitosis events. This function first finds a time point t where there are two daughter cells, and then splits the sample into two trajectories, one for each daughter cell. For cell mapping, this function uses centroid Euclidean distance. This function assumes that the input sample contains at least one mitosis event, meaning after some time point t, there should be two daughter cells; before time t, there should be only one cell at each time point.

    Args:
        sample (List[SingleCellStatic]): The input sample of single-cell static objects. This sample should contain at least one mitosis even, meaning after some time point t, there should be two daughter cells; before time t, there should be only one cell at each time point .

    Returns:
        Union[List[List[SingleCellStatic]], Tuple[List[List[SingleCellStatic]], int]: A list of trajectories, where each trajectory is a list of SingleCellStatic objects. If return_break_idx is True, also return the index of the time point t where the mitosis event happens.

    Raises:
        None.

    """
    time2scs = get_time2scs(sample)
    times = sorted(time2scs.keys())
    break_time = None
    break_time_idx = None
    common_track = []
    for idx, time in enumerate(times):
        if len(time2scs[time]) > 1:
            break_time = time
            break_time_idx = idx
            break
        else:
            common_track.append(time2scs[time][0])
    if break_time is None:
        # None is important here: it means no mitosis event and the sample is not split
        return [sample], None

    # Generate trajectories
    cur_trajs = [common_track]
    for idx, time in enumerate(times[break_time_idx:]):
        # Corner case, some time after break time contains only 1 sc
        cur_scs = time2scs[time]
        if len(cur_scs) == 1:
            cur_trajs = [traj + cur_scs for traj in cur_trajs]
            continue
        else:
            # Simply match scs according to distance
            new_tracks = []
            for sc in cur_scs:
                min_dist = float("inf")
                min_traj = None
                for traj in cur_trajs:
                    last_sc = traj[-1]
                    dist = np.linalg.norm(sc.get_center(crop=False) - last_sc.get_center(crop=False))
                    if dist < min_dist:
                        min_dist = dist
                        min_traj = traj
                if min_traj is None:
                    main_warning("Cannot find a matching traj for sc", sc)
                else:
                    new_tracks.append(min_traj + [sc])
            cur_trajs = new_tracks

    if return_break_idx:
        return cur_trajs, break_time_idx
    else:
        return cur_trajs


def make_pseudo_square_contour(sc, x=None, y=None, bbox_size=20, dim_thresholds=None):
    if x is None or y is None:
        regionprop = sc.compute_regionprops(crop=False)
        # Get the coordinates of the region
        coords = regionprop.coords
        # Calculate the geometric center (mean of the coordinates)
        geometric_center = coords.mean(axis=0)
        # Round the geometric center to get integer coordinates
        geometric_center_rounded = np.round(geometric_center).astype(int)

        # x, y = regionprop.centroid
        x, y = geometric_center_rounded

    # contour = np.array([[x-5, y-5], [x+5, y-5], [x+5, y+5], [x-5, y+5]], dtype=int)
    contour = np.array(
        [
            [x - bbox_size, y - bbox_size],
            [x + bbox_size, y - bbox_size],
            [x + bbox_size, y + bbox_size],
            [x - bbox_size, y + bbox_size],
        ],
        dtype=int,
    )
    if dim_thresholds is not None:
        contour[:, 0] = np.clip(contour[:, 0], 0, dim_thresholds[0] - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, dim_thresholds[1] - 1)
    sc.update_contour(contour)
    return sc


def make_pseudo_square_contour_sct(sct: SingleCellTrajectory, bbox_size=20):
    for t, sc in sct:
        make_pseudo_square_contour(sc, None, None, bbox_size=bbox_size)
    return sct


def make_pseudo_square_contour_wrapper(sc, dim_thresholds=None):
    make_pseudo_square_contour(sc, dim_thresholds=dim_thresholds)
    return sc
