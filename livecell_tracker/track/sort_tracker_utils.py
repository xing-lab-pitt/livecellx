from livecell_tracker.core.single_cell import SingleCellStatic, SingleCellTrajectory


def convert_sort_bbox_results_to_single_cell_trajs(all_track_bboxes, raw_img_dataset):
    id_to_sc_trajs = {}
    for timeframe, objects in enumerate(all_track_bboxes):
        for obj in objects:
            track_id = obj[-1]
            if not (track_id in id_to_sc_trajs):
                new_traj = SingleCellTrajectory(raw_img_dataset, track_id=track_id)
                id_to_sc_trajs[track_id] = new_traj
            # print("obj: ", obj)
            sc = SingleCellStatic(
                timeframe, bbox=obj[:4], img_dataset=raw_img_dataset
            )  # final column is track_id, ignore as we only need bbox here
            _traj = id_to_sc_trajs[track_id]
            _traj.add_timeframe_data(timeframe, sc)
    return id_to_sc_trajs
