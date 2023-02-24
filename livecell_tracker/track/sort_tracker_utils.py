from typing import Dict, List
from livecell_tracker.core.datasets import LiveCellImageDataset
from livecell_tracker.core.single_cell import (
    SingleCellStatic,
    SingleCellTrajectory,
    SingleCellTrajectoryCollection,
)

import numpy as np

from livecell_tracker.track.sort_tracker import Sort

SORT_EMPTY_TIMEFRAME_BBOX_DATA = np.empty((0, 5))


def is_SORT_tracker_result_empty(sort_tracker):
    return sort_tracker.time_since_update is not None


def convert_sort_bbox_results_to_single_cell_trajs(all_track_bboxes, raw_img_dataset):
    """convert raw bbox tracking results from SORT to a dictionary

    Parameters
    ----------
    all_track_bboxes : _type_
        _description_
    raw_img_dataset : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    id_to_sc_trajs = {}
    for timeframe, objects in enumerate(all_track_bboxes):
        for obj in objects:
            track_id = obj[-1]
            if not (track_id in id_to_sc_trajs):
                new_traj = SingleCellTrajectory(raw_img_dataset, track_id=track_id)
                id_to_sc_trajs[track_id] = new_traj
            # final column is track_id, ignore as we only need bbox here
            sc = SingleCellStatic(timeframe, bbox=obj[:4], img_dataset=raw_img_dataset)
            _traj = id_to_sc_trajs[track_id]
            _traj.add_single_cell(timeframe, sc)
    return id_to_sc_trajs


def get_bbox_from_contour(contour: list) -> np.array:
    """get bboxes from a contour

    Parameters
    ----------
    contour : list
        a list of (x, y) points, with shape #pts x 2
    Returns
    -------
        bounding box of the input contour, with length=4
    """
    contour = np.array(contour)
    return np.array(
        [
            contour[:, 0].min(),
            contour[:, 1].min(),
            contour[:, 0].max(),
            contour[:, 1].max(),
        ]
    )


def gen_SORT_detections_input_from_contours(contours):
    """
    generate detections for SORT tracker. detections: [x1, y1, x2, y2, score]
    ----------
    label_mask :
        an image
    Returns
    -------
    A list of (x1, y1, x2, y2, score]) for each object detected
    """
    contour_bbs = [get_bbox_from_contour(contour) for contour in contours]
    detections = np.array([list(bbox) + [1] for bbox in contour_bbs])
    return detections, contour_bbs


def map_SORT_detections_to_contour_bbs(track_bbs, contour_bbs, contours):
    detection_contours = []
    # TODO: optimize later
    for det in track_bbs:
        det_bbs = det[4:8]  # 4-8 contains original bbox
        for idx, contour_bb in enumerate(contour_bbs):
            if np.allclose(det_bbs, contour_bb, atol=1e-5):
                detection_contours.append(contours[idx])
                break
        else:
            raise Exception("fail to find contour for some detection: " + (str(det_bbs)))
    return detection_contours


def update_traj_collection_by_SORT_tracker_detection(
    traj_collection: SingleCellTrajectoryCollection,
    timeframe,
    track_bbs,
    contours,
    contour_bbs,
    raw_img_dataset: LiveCellImageDataset = None,
    sc_kwargs=dict(),
):
    det_contours = map_SORT_detections_to_contour_bbs(track_bbs, contour_bbs, contours)
    for idx, det in enumerate(track_bbs):
        track_id = det[-1]  # track_id is the last element in the detection from SORT
        if not (track_id in traj_collection):
            new_traj = SingleCellTrajectory(track_id=track_id, img_dataset=raw_img_dataset)
            traj_collection.add_trajectory(new_traj)

        sc = SingleCellStatic(
            timeframe,
            bbox=[
                det[0],
                det[1],
                det[2] + 1,
                det[3] + 1,
            ],  # Note: definition of bbox is different from det here, so +1 is necessary
            img_dataset=raw_img_dataset,
            contour=det_contours[idx],
            **sc_kwargs,
        )  # final column is track_id, ignore as we only need bbox here
        sc.update_bbox()  # further prevent from bbox diffinition differences
        _traj = traj_collection.get_trajectory(track_id)
        _traj.add_single_cell(timeframe, sc)


def track_SORT_bbox_from_contours(
    path2contours: Dict[str, np.array],
    raw_imgs: LiveCellImageDataset,
    max_age=5,
    min_hits=3,
    sc_kwargs=dict(),
):
    tracker = Sort(max_age=max_age, min_hits=min_hits)
    traj_collection = SingleCellTrajectoryCollection()
    all_track_bbs = []
    for time in raw_imgs.time2url:
        print("matching image path:", raw_imgs.get_img_path(time))
        img_path = raw_imgs.get_img_path(time)
        # TODO: fix in the future only for windows... somehow json lib saved double slashes
        contours = path2contours[raw_imgs.get_img_path(time)]["contours"]

        # TODO: for RPN based models, we may directly get bboxes from the model outputs
        detections, contour_bbs = gen_SORT_detections_input_from_contours(contours)
        track_bbs_ids = tracker.update(detections, ret_origin_bbox=True)
        # print(track_bbs_ids)
        all_track_bbs.append(track_bbs_ids)
        update_traj_collection_by_SORT_tracker_detection(
            traj_collection,
            time,
            track_bbs_ids,
            contours,
            contour_bbs,
            raw_img_dataset=raw_imgs,
            sc_kwargs=sc_kwargs,
        )
    return traj_collection


def track_SORT_bbox_from_scs(
    single_cells: List[SingleCellStatic],
    raw_imgs: LiveCellImageDataset,
    mask_dataset: LiveCellImageDataset = None,
    max_age=5,
    min_hits=3,
):
    url2contours = {}
    for sc in single_cells:
        timeframe = sc.timeframe
        if not (raw_imgs.get_img_path(timeframe) in url2contours):
            url2contours[raw_imgs.get_img_path(timeframe)] = {
                "contours": [],
            }
        url2contours[raw_imgs.get_img_path(timeframe)]["contours"].append(sc.contour)
    sc_kwargs = {
        "mask_dataset": mask_dataset,
    }
    return track_SORT_bbox_from_contours(url2contours, raw_imgs, max_age, min_hits, sc_kwargs=sc_kwargs)
