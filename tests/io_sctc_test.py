import unittest
import os
import json

from pathlib import Path
import numpy as np
from deepdiff import DeepDiff
from livecell_tracker import sample_data
from livecell_tracker.core.sc_key_manager import SingleCellMetaKeyManager as SCKM
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
from test_utils import TestHelper


class SingleCellTrajectoryCollectionIOTest(TestHelper):
    @classmethod
    def setUpClass(cls):
        # this is a set up method to create a sample SingleCellTrajectory object
        # and any necessary data for the tests
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)

        cls.traj_collection = track_SORT_bbox_from_scs(
            single_cells, dic_dataset, mask_dataset=mask_dataset, max_age=1, min_hits=1
        )

    def setUp(self):
        self.io_out_dir = Path("test_io_output")
        self.io_out_dir.mkdir(exist_ok=True)  # Make sure the directory exists before each test
        self.json_file_path = self.io_out_dir / "test_sct_collection.json"
        self.helper = TestHelper()

    def _test_write_dataset(self, dataset, dataset_json_path_key, json_dict=None):
        if dataset is not None and dataset_json_path_key in json_dict:
            dir_path = os.path.dirname(json_dict[dataset_json_path_key])  # get the directory path
            if not os.path.exists(dir_path):  # check if the directory exists
                os.makedirs(dir_path)  # create the directory
            with open(json_dict[dataset_json_path_key], "w+") as f:
                json.dump(dataset.to_json_dict(), f)

    def test_to_json_dict(self):
        json_dict = self.traj_collection.to_json_dict()
        self.assertIsInstance(json_dict, dict)
        self.assertIn("track_id_to_trajectory", json_dict)

        traj_dict = json_dict["track_id_to_trajectory"]
        self.assertIsInstance(traj_dict, dict)
        self.assertEqual(len(traj_dict), len(self.traj_collection))

        for track_id, traj in traj_dict.items():
            traj_instance = self.traj_collection.get_trajectory(track_id)
            self._test_write_dataset(traj_instance.img_dataset, "img_dataset_json_path", traj)
            self._test_write_dataset(traj_instance.mask_dataset, "mask_dataset_json_path", traj)

            self.assertIsInstance(track_id, int)
            self.assertIsInstance(traj, dict)

            # Check if the returned trajectory has correct format
            print("Loading dataset from", traj["img_dataset_json_path"])  # print file path
            loaded_sct = SingleCellTrajectory().load_from_json_dict(traj)
            original_sct = self.traj_collection.get_trajectory(track_id)
            self.helper.assertEqualSCTs(original_sct, loaded_sct)

    def test_load_from_json_dict(self):
        json_dict = self.traj_collection.to_json_dict(self.io_out_dir)
        for track_id, traj in self.traj_collection.track_id_to_trajectory.items():
            # Convert the SingleCellTrajectory object to a JSON dict
            traj_json_dict = traj.to_json_dict(self.io_out_dir)
            # Write the datasets to files
            print("json_dict contains img_dataset_json_path:", "img_dataset_json_path" in traj_json_dict)
            print("json_dict contains mask_dataset_json_path:", "mask_dataset_json_path" in traj_json_dict)
            self._test_write_dataset(traj.img_dataset, "img_dataset_json_path", traj_json_dict)
            self._test_write_dataset(traj.mask_dataset, "mask_dataset_json_path", traj_json_dict)

        loaded_collection = SingleCellTrajectoryCollection().load_from_json_dict(json_dict)

        self.assertIsInstance(loaded_collection, SingleCellTrajectoryCollection)
        self.assertEqual(len(loaded_collection), len(self.traj_collection))

        # Check each SingleCellTrajectory
        for track_id, original_sct in self.traj_collection.track_id_to_trajectory.items():
            loaded_sct = loaded_collection.get_trajectory(track_id)
            self.helper.assertEqualSCTs(loaded_sct, original_sct)

    def test_write_json(self):
        self.traj_collection.write_json(path=self.json_file_path, dataset_json_dir=self.io_out_dir)
        self.assertTrue(os.path.exists(self.json_file_path))

        with open(self.json_file_path, "r") as f:
            json_dict = json.load(f)

        loaded_collection = SingleCellTrajectoryCollection().load_from_json_dict(json_dict)
        self.assertEqual(len(loaded_collection), len(self.traj_collection))

        # Check each SingleCellTrajectory
        for track_id, original_sct in self.traj_collection.track_id_to_trajectory.items():
            loaded_sct = loaded_collection.get_trajectory(track_id)
            self.helper.assertEqualSCTs(loaded_sct, original_sct)

    def test_load_from_json_file(self):
        self.traj_collection.write_json(path=self.json_file_path, dataset_json_dir=self.io_out_dir)
        print("Write json file using write_json from test_load_from_json_file in current directory:", os.getcwd())
        new_collection = SingleCellTrajectoryCollection().load_from_json_file(path=self.json_file_path)

        self.assertIsInstance(new_collection, SingleCellTrajectoryCollection)
        self.assertEqual(len(new_collection), len(self.traj_collection))

        # Check each SingleCellTrajectory
        for track_id, original_sct in self.traj_collection.track_id_to_trajectory.items():
            new_sct = new_collection.get_trajectory(track_id)
            self.helper.assertEqualSCTs(new_sct, original_sct)

    def tearDown(self):
        # Remove test file after the tests run
        if os.path.exists(self.json_file_path):
            os.remove(self.json_file_path)


if __name__ == "__main__":
    unittest.main()
