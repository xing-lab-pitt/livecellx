import glob
import os
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm
from skimage import measure


def get_contours_from_pred_masks(instance_pred_masks):
    # TODO add docs later
    contours = []
    for instance_mask in instance_pred_masks:
        tmp_contours = measure.find_contours(
            instance_mask, level=0.5, fully_connected="low", positive_orientation="low"
        )
        if len(tmp_contours) != 1:
            print("[WARN] more than 1 contour found in the instance mask")
        # convert to list for saving into json
        contours.extend([[list(coords) for coords in coord_arr] for coord_arr in tmp_contours])
    return contours
