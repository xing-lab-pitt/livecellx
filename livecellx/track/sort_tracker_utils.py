from typing import Dict, List

import tqdm
from livecellx.core.datasets import LiveCellImageDataset
from livecellx.core.single_cell import (
    SingleCellStatic,
    SingleCellTrajectory,
    SingleCellTrajectoryCollection,
)

import numpy as np
from livecellx.livecell_logger import main_warning

from livecellx.track.sort_tracker import Sort

SORT_EMPTY_TIMEFRAME_BBOX_DATA = np.empty((0, 5))


def is_SORT_tracker_result_empty(sort_tracker):
    return sort_tracker.time_since_update is not None


def get_bbox_from_contour(contour: list) -> np.ndarray:
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
    scs_at_t=None,
    sc_inplace=False,
):
    def _match_sc_by_bbox(bbox, scs, atol=10):
        for tmp_sc in scs:
            if np.allclose(bbox, tmp_sc.bbox, atol=atol):
                return tmp_sc
        return None

    if sc_inplace:
        assert scs_at_t is not None, "scs must be provided if sc_inplace is True"
    det_contours = map_SORT_detections_to_contour_bbs(track_bbs, contour_bbs, contours)
    for idx, det in enumerate(track_bbs):
        track_id = det[-1]  # track_id is the last element in the detection from SORT
        if not (track_id in traj_collection):
            new_traj = SingleCellTrajectory(track_id=track_id, img_dataset=raw_img_dataset)
            traj_collection.add_trajectory(new_traj)
        # bbox = [
        #         det[0],
        #         det[1],
        #         det[2] + 1,
        #         det[3] + 1,
        # ]
        bbox = det[4:8]  # Use the original bbox from scs
        cid = None
        matched_old_sc = None
        if scs_at_t is not None:
            matched_old_sc = _match_sc_by_bbox(bbox, scs_at_t)
            cid = matched_old_sc.id if matched_old_sc is not None else None
            if cid is None:
                main_warning("[Tracking by SORT] fail to find re-matched sc for bbox: " + str(bbox))
        sc = None
        if (not sc_inplace) or (matched_old_sc is None):
            if sc_inplace and matched_old_sc is None:
                main_warning(f"[Tracking by SORT] fail to find re-matched sc for bbox: {bbox} at time: {timeframe}")
            sc = SingleCellStatic(
                id=cid,
                timeframe=timeframe,
                bbox=bbox,  # Note: definition of skimage bbox is different from det here, so +1 is necessary
                img_dataset=raw_img_dataset,
                contour=det_contours[idx],
                **sc_kwargs,
            )  # final column is track_id, ignore as we only need bbox here
        else:
            sc = matched_old_sc
        sc.update_bbox()  # further prevent from bbox diffinition differences
        _traj = traj_collection.get_trajectory(track_id)
        _traj.add_single_cell_by_time(timeframe, sc)


def track_SORT_bbox_from_contours(
    time2contours: Dict[str, np.array],
    raw_imgs: LiveCellImageDataset,
    max_age=5,
    min_hits=3,
    sc_kwargs=dict(),
    scs=None,
    sc_inplace=False,
):
    """
    Tracks bounding boxes using the SORT algorithm based on contours.

    Args:
        time2contours (Dict[str, np.array]): A dictionary mapping timeframes to contours.
        raw_imgs (LiveCellImageDataset): The raw image dataset.
        max_age (int, optional): The maximum age of a track before it is deleted. Defaults to 5.
        min_hits (int, optional): The minimum number of hits required for a track to be considered valid. Defaults to 3.
        sc_kwargs (dict, optional): Additional keyword arguments for single cell tracking. Defaults to an empty dictionary.
        scs (list, optional): A list of single cells. Defaults to None.
        sc_inplace (bool, optional): Whether to update the single cell objects in-place. Defaults to False.

    Returns:
        SingleCellTrajectoryCollection: The collection of single cell trajectories.
    """
    tracker = Sort(max_age=max_age, min_hits=min_hits)
    traj_collection = SingleCellTrajectoryCollection()
    all_track_bbs = []
    sorted_times = sorted(time2contours.keys())
    for timeframe in tqdm.tqdm(sorted_times, desc="SORT tracking..."):
        # TODO: fix in the future only for windows... somehow json lib saved double slashes
        contours = time2contours[timeframe]["contours"]

        # TODO: for RPN based models, we may directly get bboxes from the model outputs
        detections, contour_bbs = gen_SORT_detections_input_from_contours(contours)
        track_bbs_ids = tracker.update(detections, ret_origin_bbox=True)
        # print(track_bbs_ids)
        all_track_bbs.append(track_bbs_ids)

        scs_at_time = [sc for sc in scs if sc.timeframe == timeframe] if scs is not None else None
        update_traj_collection_by_SORT_tracker_detection(
            traj_collection,
            timeframe,
            track_bbs_ids,
            contours,
            contour_bbs,
            raw_img_dataset=raw_imgs,
            sc_kwargs=sc_kwargs,
            scs_at_t=scs_at_time,
            sc_inplace=sc_inplace,
        )
    return traj_collection


def track_SORT_bbox_from_scs(
    single_cells: List[SingleCellStatic],
    raw_imgs: LiveCellImageDataset,
    mask_dataset: LiveCellImageDataset = None,
    max_age=5,
    min_hits=3,
    sc_inplace=False,
):
    """
    Tracks the bounding boxes of single cells using the SORT algorithm based on the given single cell objects.

    Args:
        single_cells (List[SingleCellStatic]): A list of SingleCellStatic objects representing the single cells to track.
        raw_imgs (LiveCellImageDataset): The raw image dataset containing the frames for tracking.
        mask_dataset (LiveCellImageDataset, optional): The mask image dataset used for segmentation. Defaults to None.
        max_age (int, optional): The maximum number of frames a track can be inactive before it is deleted. Defaults to 5.
        min_hits (int, optional): The minimum number of hits (overlapping detections) required to initiate a track. Defaults to 3.
        sc_inplace (bool, optional): Whether to modify the SingleCellStatic objects in-place or generate new single cell objects. Defaults to False.

    Returns:
        List[SingleCellStatic]: A list of newly generated or existing SingleCellStatic objects, based on args, with updated bounding box coordinates.
    """

    time2contours = {}
    for sc in single_cells:
        timeframe = sc.timeframe
        if not timeframe in time2contours:
            time2contours[timeframe] = {
                "contours": [],
            }
        time2contours[timeframe]["contours"].append(sc.contour)
    sc_kwargs = {
        "mask_dataset": mask_dataset,
    }
    return track_SORT_bbox_from_contours(
        time2contours, raw_imgs, max_age, min_hits, sc_kwargs=sc_kwargs, scs=single_cells, sc_inplace=sc_inplace
    )
