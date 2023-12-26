from typing import List
import numpy as np
from livecellx.core.single_cell import (
    SingleCellStatic,
    SingleCellTrajectory,
    SingleCellTrajectoryCollection,
    get_time2scs,
)
from livecellx.livecell_logger import main_warning


def split_mitosis_sample(sample: List[SingleCellStatic]) -> List[List[SingleCellStatic]]:
    """
    Splits a list of single-cell static objects into multiple trajectories based on mitosis events. This function first finds a time point t where there are two daughter cells, and then splits the sample into two trajectories, one for each daughter cell. For cell mapping, this function uses centroid Euclidean distance. This function assumes that the input sample contains at least one mitosis event, meaning after some time point t, there should be two daughter cells; before time t, there should be only one cell at each time point.

    Args:
        sample (List[SingleCellStatic]): The input sample of single-cell static objects. This sample should contain at least one mitosis even, meaning after some time point t, there should be two daughter cells; before time t, there should be only one cell at each time point .

    Returns:
        List[List[SingleCellStatic]]: A list of trajectories, where each trajectory represents a separate mitosis event.

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
        return [sample]

    # Generate trajectories
    cur_trajs = [common_track]
    for idx, time in enumerate(times[break_time_idx:]):
        # Corner case, some time after break time contains only 1 sc
        cur_scs = time2scs[time]
        if len(cur_scs) == 1:
            cur_trajs = [traj + cur_scs[0] for traj in cur_trajs]
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
    return cur_trajs
