import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.io import imread
import glob
from pathlib import Path
from PIL import Image, ImageSequence
from tqdm import tqdm
import os, cv2
import os.path
import pandas as pd

# from livecellx import segment
from livecellx.track.sort_tracker_utils import (
    gen_SORT_detections_input_from_contours,
    update_traj_collection_by_SORT_tracker_detection,
    track_SORT_bbox_from_contours,
    track_SORT_bbox_from_scs,
)
from livecellx import sample_data
from livecellx import core
from livecellx.core import datasets, pl_utils
from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset
from livecellx.segment.utils import prep_scs_from_mask_dataset
from livecellx.preprocess.correct_bg import correct_background_bisplrep, correct_background_median_gamma
from livecellx.preprocess.utils import normalize_img_to_uint8
from livecellx.trajectory.feature_extractors import compute_skimage_regionprops

from livecellx.core import SingleCellTrajectory, SingleCellStatic, SingleCellTrajectoryCollection


import pytest
import unittest
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure
from skimage.measure import regionprops

import unittest
import matplotlib
import matplotlib.pyplot as plt


class TestTutorialGeneralUse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # load the single_cells here, this code will run once before all tests
        cls.dic_dataset, cls.mask_dataset = sample_data.tutorial_three_image_sys()
        cls.single_cells = prep_scs_from_mask_dataset(cls.mask_dataset, cls.dic_dataset)

    def test_loading_single_cells(self):
        # Verify that each single cell has a mask dataset
        for sc in self.single_cells:
            assert sc.mask_dataset is not None

        assert len(self.single_cells) == 42

        single_cells_by_time = {}
        for cell in self.single_cells:
            if cell.timeframe not in single_cells_by_time:
                single_cells_by_time[cell.timeframe] = []
            single_cells_by_time[cell.timeframe].append(cell)

        assert len(single_cells_by_time[0]) == 13
        assert len(single_cells_by_time[1]) == 14
        assert len(single_cells_by_time[2]) == 15

    def test_visualize_single_cell(self):
        sc = self.single_cells[0]
        assert isinstance(sc, SingleCellStatic)

        # Create subplots
        fig, axes = plt.subplots(1, 4, figsize=(10, 5))
        assert fig is not None
        assert axes is not None

        # TODO: Compare the visualizations to the expected output
        # Show single cell
        try:
            sc.show(ax=axes[0])
            sc.show_mask(ax=axes[1])
            sc.show_contour_img(ax=axes[2])
            sc.show_contour_mask(ax=axes[3])
        except Exception as e:
            assert False, f"An error occurred while visualizing single cell: {str(e)}"

        # Show panel
        try:
            sc.show_panel(figsize=(15, 5))
        except Exception as e:
            assert False, f"An error occurred while showing panel: {str(e)}"

    def test_preprocess_bg_correct_and_visualize(self):
        # Preprocess
        sc = self.single_cells[0].copy()
        padding_size = 30
        sc_img = sc.get_img_crop(padding=padding_size)
        bisplrep_sc_img = normalize_img_to_uint8(sc_img)
        bisplrep_sc_img = correct_background_bisplrep(sc_img, sample_step=5, s=1e20)
        gamma_sc_img = sc.get_img_crop(padding=padding_size)
        gamma_sc_img = normalize_img_to_uint8(sc_img)
        gamma_sc_img = correct_background_median_gamma(sc_img, disk_size=2)

        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        im = axes[0].imshow(sc_img)
        pl_utils.add_colorbar(im, axes[0], fig)
        axes[0].set_title("original")
        im = axes[1].imshow(bisplrep_sc_img)
        pl_utils.add_colorbar(im, axes[1], fig)
        axes[1].set_title("corrected: bisplrep")
        im = axes[2].imshow(gamma_sc_img)
        pl_utils.add_colorbar(im, axes[2], fig)
        axes[2].set_title("corrected: gamma correction")

        # Test run without errors
        self.assertTrue(True)

    def test_cell_features_and_overlap(self):
        sc1 = self.single_cells[1]
        sc2 = self.single_cells[2]

        # Calculate cell features
        skimage_features = compute_skimage_regionprops(sc1)
        sc1.add_feature("skimage", skimage_features)

        # Assert features have been correctly added
        feature_series = sc1.get_feature_pd_series()
        assert isinstance(feature_series, pd.Series)
        assert not feature_series.empty
        assert any(idx.startswith("skimage") for idx in feature_series.index)

        # test for bounding box
        bbox = sc2.bbox
        assert len(bbox) == 4
        assert isinstance(bbox, np.ndarray)

        # Calculate overlap between two single cells
        iou, overlap_percent = sc1.compute_iou(sc2), sc1.compute_overlap_percent(sc2)

        assert isinstance(iou, float)
        assert isinstance(overlap_percent, float)
        # IOU and overlap_percent should be 0 because the two single cells are not overlapping
        assert iou == 0
        assert 0 == overlap_percent

    def test_tracking_based_on_single_cells(self):
        traj_collection = track_SORT_bbox_from_scs(
            self.single_cells, self.dic_dataset, mask_dataset=self.mask_dataset, max_age=1, min_hits=1
        )

        # Assert traj_collection is an instance of SingleCellTrajectoryCollection
        assert isinstance(
            traj_collection, SingleCellTrajectoryCollection
        ), "Returned value is not an instance of SingleCellTrajectoryCollection"

        ax = traj_collection.histogram_traj_length()

        # Assert the return value of histogram_traj_length is an instance of Axes
        assert isinstance(ax, matplotlib.axes.Axes), "Returned value is not an instance of matplotlib.axes.Axes"


if __name__ == "__main__":
    unittest.main()
