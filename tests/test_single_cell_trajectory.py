import unittest
import numpy as np
from pathlib import Path
import tempfile
import os
from livecellx import sample_data
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.core.single_cell import SingleCellStatic, SingleCellTrajectory
from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset
from tests.test_utils import TestHelper


class SingleCellTrajectoryTestCase(TestHelper):
    @classmethod
    def setUpClass(cls):
        # Load sample data
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        cls.single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)
        cls.img_dataset = dic_dataset
        cls.mask_dataset = mask_dataset

    def setUp(self):
        # Create a trajectory with a few cells
        self.trajectory = SingleCellTrajectory(track_id=1, img_dataset=self.img_dataset, mask_dataset=self.mask_dataset)

        # Add cells to the trajectory
        for i, sc in enumerate(self.single_cells[:3]):
            sc_copy = sc.copy()
            sc_copy.timeframe = i  # Ensure different timeframes
            self.trajectory.add_sc(sc_copy)

    def test_init(self):
        """Test the initialization of a SingleCellTrajectory instance"""
        sct = SingleCellTrajectory(track_id=2)
        self.assertEqual(sct.track_id, 2)
        self.assertEqual(len(sct.timeframe_to_single_cell), 0)
        self.assertEqual(len(sct.mother_trajectories), 0)
        self.assertEqual(len(sct.daughter_trajectories), 0)

    def test_compute_features(self):
        """Test computing features for all cells in the trajectory"""
        # Define a simple feature computation function
        def compute_area(sc):
            return np.array([100])  # Return a numpy array as required

        # Compute features
        self.trajectory.compute_features("area", compute_area)

        # Check that all cells have the feature
        for _, sc in self.trajectory:
            self.assertIn("area", sc.feature_dict)
            self.assertEqual(sc.feature_dict["area"][0], 100)

    def test_timeframe_set_property(self):
        """Test the timeframe_set property"""
        expected_timeframes = {0, 1, 2}
        self.assertEqual(self.trajectory.timeframe_set, expected_timeframes)

    def test_times_property(self):
        """Test the times property"""
        expected_times = [0, 1, 2]
        self.assertEqual(self.trajectory.times, expected_times)

    def test_add_sc_by_time(self):
        """Test adding a single cell by timeframe"""
        # Create a new cell
        sc = self.single_cells[3].copy()
        sc.timeframe = 3

        # Add the cell to the trajectory
        self.trajectory.add_sc_by_time(3, sc)

        # Check that the cell was added
        self.assertIn(3, self.trajectory.timeframe_to_single_cell)
        self.assertEqual(self.trajectory.timeframe_to_single_cell[3], sc)

    def test_add_sc(self):
        """Test adding a single cell"""
        # Create a new cell
        sc = self.single_cells[3].copy()
        sc.timeframe = 4

        # Add the cell to the trajectory
        self.trajectory.add_sc(sc)

        # Check that the cell was added
        self.assertIn(4, self.trajectory.timeframe_to_single_cell)
        self.assertEqual(self.trajectory.timeframe_to_single_cell[4], sc)

    def test_get_img(self):
        """Test getting an image for a specific timeframe"""
        # Get the image for timeframe 0
        img = self.trajectory.get_img(0)

        # Check that the image is not None
        self.assertIsNotNone(img)
        self.assertTrue(isinstance(img, np.ndarray))

    def test_get_mask(self):
        """Test getting a mask for a specific timeframe"""
        # Get the mask for timeframe 0
        mask = self.trajectory.get_mask(0)

        # Check that the mask is not None
        self.assertIsNotNone(mask)
        self.assertTrue(isinstance(mask, np.ndarray))

    def test_get_timeframe_span(self):
        """Test getting the timeframe span"""
        # The trajectory has cells at timeframes 0, 1, 2
        expected_span = (0, 2)
        self.assertEqual(self.trajectory.get_timeframe_span(), expected_span)

    def test_get_timeframe_span_length(self):
        """Test getting the timeframe span length"""
        # The trajectory has cells at timeframes 0, 1, 2
        expected_length = 3  # 2 - 0 + 1
        self.assertEqual(self.trajectory.get_timeframe_span_length(), expected_length)

    def test_get_sc(self):
        """Test getting a single cell for a specific timeframe"""
        # Get the cell for timeframe 1
        sc = self.trajectory.get_sc(1)

        # Check that the cell is not None and has the correct timeframe
        self.assertIsNotNone(sc)
        self.assertEqual(sc.timeframe, 1)

    def test_to_json_dict(self):
        """Test converting a SingleCellTrajectory to a JSON dictionary"""
        # Create a temporary directory for dataset JSON files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert the trajectory to a JSON dictionary
            json_dict = self.trajectory.to_json_dict(dataset_json_dir=temp_dir)

            # Check that the JSON dictionary has the expected fields
            self.assertEqual(json_dict["track_id"], 1)
            self.assertIn("timeframe_to_single_cell", json_dict)
            self.assertEqual(len(json_dict["timeframe_to_single_cell"]), 3)
            self.assertIn("meta", json_dict)

    def test_load_from_json_dict(self):
        """Test loading a SingleCellTrajectory from a JSON dictionary"""
        # Create a temporary directory for dataset JSON files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert the trajectory to a JSON dictionary
            json_dict = self.trajectory.to_json_dict(dataset_json_dir=temp_dir)

            # Create a new trajectory and load from the JSON dictionary
            new_trajectory = SingleCellTrajectory()
            new_trajectory.load_from_json_dict(json_dict, img_dataset=self.img_dataset)

            # Check that the new trajectory has the expected fields
            self.assertEqual(new_trajectory.track_id, 1)
            self.assertEqual(len(new_trajectory.timeframe_to_single_cell), 3)
            self.assertEqual(new_trajectory.timeframe_set, {0, 1, 2})

    def test_copy(self):
        """Test copying a SingleCellTrajectory"""
        # Copy the trajectory
        copied_trajectory = self.trajectory.copy()

        # Check that the copy has the same fields
        self.assertEqual(copied_trajectory.track_id, self.trajectory.track_id)
        self.assertEqual(len(copied_trajectory.timeframe_to_single_cell), len(self.trajectory.timeframe_to_single_cell))
        self.assertEqual(copied_trajectory.timeframe_set, self.trajectory.timeframe_set)

        # Check that the copy is a different object
        self.assertIsNot(copied_trajectory, self.trajectory)

        # Test copy with copy_scs=True
        copied_trajectory_with_scs = self.trajectory.copy(copy_scs=True)

        # Check that the single cells are also copied
        for timeframe, sc in self.trajectory.timeframe_to_single_cell.items():
            copied_sc = copied_trajectory_with_scs.timeframe_to_single_cell[timeframe]
            self.assertIsNot(copied_sc, sc)
            self.assertEqual(copied_sc.timeframe, sc.timeframe)

    def test_is_empty(self):
        """Test checking if a SingleCellTrajectory is empty"""
        # The trajectory has cells, so it should not be empty
        self.assertFalse(self.trajectory.is_empty())

        # Create an empty trajectory
        empty_trajectory = SingleCellTrajectory()

        # The empty trajectory should be empty
        self.assertTrue(empty_trajectory.is_empty())

    def test_get_prev_by_sc(self):
        """Test getting the previous single cell for a given single cell"""
        # Get the cell for timeframe 2
        sc = self.trajectory.get_sc(2)

        # Get the previous cell
        prev_sc = self.trajectory.get_prev_by_sc(sc)

        # Check that the previous cell has timeframe 1
        self.assertIsNotNone(prev_sc)
        self.assertEqual(prev_sc.timeframe, 1)

        # Get the cell for timeframe 0
        sc = self.trajectory.get_sc(0)

        # Get the previous cell (should be None)
        prev_sc = self.trajectory.get_prev_by_sc(sc)

        # Check that there is no previous cell
        self.assertIsNone(prev_sc)

    def test_add_nonoverlapping_sct(self):
        """Test adding a non-overlapping SingleCellTrajectory"""
        # Create a new trajectory with cells at timeframes 3, 4, 5
        other_trajectory = SingleCellTrajectory(track_id=2)
        for i in range(3, 6):
            sc = self.single_cells[0].copy()
            sc.timeframe = i
            other_trajectory.add_sc(sc)

        # Add the other trajectory to the original trajectory
        self.trajectory.add_nonoverlapping_sct(other_trajectory)

        # Check that the original trajectory now has cells at timeframes 0, 1, 2, 3, 4, 5
        expected_timeframes = {0, 1, 2, 3, 4, 5}
        self.assertEqual(self.trajectory.timeframe_set, expected_timeframes)

        # Try to add a trajectory with overlapping timeframes
        overlapping_trajectory = SingleCellTrajectory(track_id=3)
        sc = self.single_cells[0].copy()
        sc.timeframe = 2  # Overlaps with the original trajectory
        overlapping_trajectory.add_sc(sc)

        # Adding the overlapping trajectory should raise a ValueError
        with self.assertRaises(ValueError):
            self.trajectory.add_nonoverlapping_sct(overlapping_trajectory)

    def test_get_scs_napari_shapes(self):
        """Test getting Napari shapes for the single cells in the trajectory"""
        # Get the shapes for bounding boxes
        bbox_shapes = self.trajectory.get_scs_napari_shapes(bbox=True)

        # Check that there are shapes for all cells
        self.assertEqual(len(bbox_shapes), 3)

        # Get the shapes for contours
        contour_shapes = self.trajectory.get_scs_napari_shapes(bbox=False)

        # Check that there are shapes for all cells
        self.assertEqual(len(contour_shapes), 3)

        # Get the shapes and the single cells
        shapes, scs = self.trajectory.get_scs_napari_shapes(return_scs=True)

        # Check that there are shapes and cells for all cells
        self.assertEqual(len(shapes), 3)
        self.assertEqual(len(scs), 3)


if __name__ == "__main__":
    unittest.main()
