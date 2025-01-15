import glob
import os
import os.path
from pathlib import Path
from typing import List, Tuple
from collections import deque
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm
from skimage import measure
from skimage.measure import regionprops
from multiprocessing import Pool
from skimage.measure import regionprops, find_contours

from livecellx.segment.ou_simulator import find_contours_opencv

from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset
from livecellx.core.single_cell import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection


def filter_boundary_cells(
    scs: List[SingleCellStatic], dist_to_boundary=30, bbox_bounds=None, use_box_center=True
) -> List[SingleCellStatic]:
    not_boundary_scs = []
    if bbox_bounds is None:
        dim = scs[0].get_img().shape[:2]
        bbox_bounds = [0, 0, dim[0], dim[1]]
    for sc in scs:
        bbox = sc.get_bbox()
        if use_box_center and (
            (bbox[0] + bbox[2]) / 2 > bbox_bounds[0] + dist_to_boundary
            and (bbox[1] + bbox[3]) / 2 > bbox_bounds[1] + dist_to_boundary
            and (bbox[0] + bbox[2]) / 2 < bbox_bounds[2] - dist_to_boundary
            and (bbox[1] + bbox[3]) / 2 < bbox_bounds[3] - dist_to_boundary
        ):
            not_boundary_scs.append(sc)
        elif (
            bbox[0] > bbox_bounds[0] + dist_to_boundary
            and bbox[1] > bbox_bounds[1] + dist_to_boundary
            and bbox[2] < bbox_bounds[2] - dist_to_boundary
            and bbox[3] < bbox_bounds[3] - dist_to_boundary
        ):
            not_boundary_scs.append(sc)
        else:
            pass
    return not_boundary_scs


def filter_scs_by_size(scs: list, min_size=-np.inf, max_size=np.inf):
    required_scs = []
    for sc in scs:
        contour = sc.contour.astype(np.float32)
        area = cv2.contourArea(contour)
        if area >= min_size and area <= max_size:
            required_scs.append(sc)
    return required_scs


def is_sct_on_boundary(sct: SingleCellTrajectory, dist: int) -> bool:
    """
    Check if the single cell trajectory is on the boundary.
    """
    scs = sct.get_all_scs()
    in_bound_scs = filter_boundary_cells(scs, dist)
    is_some_cell_near_boundary = len(in_bound_scs) != len(scs)
    return is_some_cell_near_boundary


def filter_boundary_traj(sctc: SingleCellTrajectoryCollection, dist: int) -> SingleCellTrajectoryCollection:
    """
    Filter trajectories based on the boundary value.
    Trajectories with a length less than or equal to the boundary value are removed.
    """
    filtered_sctc = SingleCellTrajectoryCollection()
    for tid, sc_traj in sctc:
        if not is_sct_on_boundary(sc_traj, dist):
            filtered_sctc.add_trajectory(sc_traj)
    return filtered_sctc
