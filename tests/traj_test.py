import matplotlib.pyplot as plt
from livecell_tracker.trajectory.contour_utils import get_cellTool_contour_points
from livecell_tracker.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from tqdm import tqdm
import json

# TODO: fix test case
def test_contour():
    traj_collection_json_path = "../datasets/test_data/traj_analysis/track_singleCellTrajectoryCollection.json"
    traj_collection_json = json.load(open(traj_collection_json_path, "r"))
    trajectory_collection = SingleCellTrajectoryCollection().load_from_json_dict(traj_collection_json)
    traj = trajectory_collection.get_trajectory(1)
    cell_contours = get_cellTool_contour_points(traj, contour_num_points=500)

    for contour in cell_contours:
        plt.plot(contour.points[:, 0], contour.points[:, 1])
    plt.show()
