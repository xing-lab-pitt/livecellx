import os
import uuid
import json
import napari
import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import pytest
from napari.layers import Shapes
from livecell_tracker import sample_data
from livecell_tracker.core.napari_visualizer import NapariVisualizer
from livecell_tracker.segment.utils import prep_scs_from_mask_dataset
from livecell_tracker.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from livecell_tracker.track.sort_tracker_utils import (
    gen_SORT_detections_input_from_contours,
    update_traj_collection_by_SORT_tracker_detection,
    track_SORT_bbox_from_contours,
    track_SORT_bbox_from_scs,
)
from livecell_tracker.core.sct_operator import SctOperator


class SctOperatorTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Skip the entire test class
        pytest.skip("Skipping SctOperatorTest")

    @classmethod
    def setUpClass(cls):
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        cls.single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)
        cls.traj_collection = track_SORT_bbox_from_scs(
            cls.single_cells, dic_dataset, mask_dataset=mask_dataset, max_age=1, min_hits=1
        )
        cls.viewer = napari.view_image(dic_dataset.to_dask(), name="dic_image", cache=True)

    def setUp(self):
        self.shape_layer = NapariVisualizer.gen_trajectories_shapes(
            self.traj_collection, self.viewer, contour_sample_num=20
        )
        self.sct_operator = SctOperator(self.traj_collection, self.shape_layer, self.viewer)

        self.sample_dir = ".\\test_sample_dir"
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

    def test_delete_selected_sct(self):
        # Given: Initial state
        # Sample action - select a trajectory to delete
        # Grabbing the first SingleCellTrajectory
        sct_to_delete = self.traj_collection.get_all_trajectories()[0]
        track_id_to_delete = sct_to_delete.track_id

        # Simulate selecting the shapes associated with the trajectory
        select_info_list = []
        for sc in sct_to_delete.get_all_scs():
            shape_index = self.sct_operator.lookup_sc_shape_index(sc)
            if shape_index is not None:
                select_info_list.append((sct_to_delete, sc, shape_index))

        self.sct_operator.select_info = select_info_list

        # When: Deleting selected trajectory
        self.sct_operator.delete_selected_sct()

        # Then: Assertions to validate behavior
        # 1. Check if the track_id is deleted from the collection
        self.assertNotIn(track_id_to_delete, self.traj_collection.get_all_tids())

        # 2. Check if the selection info is cleared
        self.assertEqual(self.sct_operator.select_info, [])  # Assuming select_info is cleared after deletion

        # 3. Check if the shape related to the deleted trajectory has been removed from the shape_layer
        shape_track_ids = self.sct_operator.shape_layer.properties["track_id"]
        self.assertNotIn(track_id_to_delete, shape_track_ids)

    def test_save_annotations(self):
        # Given: Setup data
        # Use the first few SingleCellStatic instances for the test
        sample_cells = self.single_cells[:5]

        self.sct_operator.annotate_click_samples = {"mitosis": [{"sample": sample_cells, "sample_id": uuid.uuid4()}]}

        # When: Calling save_annotations method
        saved_sample_paths = self.sct_operator.save_annotations(self.sample_dir)

        # Then: Assertions to validate behavior

        # 1. Verify the number of saved sample paths
        self.assertEqual(len(saved_sample_paths), 1)

        # 2. Verify the saved sample paths start and end correctly
        expected_start_path = os.path.normpath(self.sample_dir)
        actual_path = os.path.normpath(str(saved_sample_paths[0]))
        self.assertTrue(actual_path.startswith(expected_start_path))
        self.assertTrue(str(saved_sample_paths[0]).endswith(".json"))

        # 3. Verify the files actually exist and contain valid JSON
        for path in saved_sample_paths:
            self.assertTrue(os.path.exists(path))
            with open(path, "r") as f:
                data = json.load(f)

                # 1. Verify the data is a list
                self.assertTrue(isinstance(data, list))

                # 2. Verify each item in the list is a dictionary representing a SingleCellStatic instance
                for item in data:
                    self.assertTrue(isinstance(item, dict))
                    # Example checks for the expected keys in each dictionary
                    for key in ["id", "timeframe", "bbox", "contour", "meta", "dataset_json_dir"]:
                        self.assertIn(key, item)

    def tearDown(self):
        # Cleanup: Delete all files and directories recursively
        if os.path.exists(self.sample_dir):
            for root, dirs, files in os.walk(self.sample_dir, topdown=False):  # topdown=False for bottom-up walking
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.sample_dir)

        # Resetting sct_operator to ensure isolation
        self.sct_operator = None


if __name__ == "__main__":
    unittest.main()
