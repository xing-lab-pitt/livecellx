import json
from livecell_tracker.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)


# TODO
def test_read_traj_collection():
    traj_collection_json_path = "../datasets/test_data/traj_analysis/test_trajs.json"
    traj_collection_json = json.load(open(traj_collection_json_path, "r"))
    trajectory_collection = SingleCellTrajectoryCollection().load_from_json_dict(traj_collection_json)

    # TODO: recursively check all the trajectories and all single cell objects
