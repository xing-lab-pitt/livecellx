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
from livecellx.core.single_cell import SingleCellStatic


def filter_boundary_cells(scs: List[SingleCellStatic], dist_to_boundary=30, bbox_bounds=None):
    not_boundary_scs = []
    if bbox_bounds is None:
        dim = scs[0].get_img().shape[:2]
        bbox_bounds = [0, 0, dim[0], dim[1]]
    for sc in scs:
        bbox = sc.get_bbox()
        if (
            bbox[0] > bbox_bounds[0] + dist_to_boundary
            and bbox[1] > bbox_bounds[1] + dist_to_boundary
            and bbox[2] < bbox_bounds[2] - dist_to_boundary
            and bbox[3] < bbox_bounds[3] - dist_to_boundary
        ):
            not_boundary_scs.append(sc)
    return not_boundary_scs


def filter_scs_by_size(scs: list, min_size=-np.inf, max_size=np.inf):
    required_scs = []
    for sc in scs:
        contour = sc.contour.astype(np.float32)
        area = cv2.contourArea(contour)
        if area >= min_size and area <= max_size:
            required_scs.append(sc)
    return required_scs
