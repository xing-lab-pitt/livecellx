import unittest
import json
import copy
import os

from pathlib import Path
import numpy as np
from deepdiff import DeepDiff
from livecellx import sample_data
from livecellx.core.sc_key_manager import SingleCellMetaKeyManager as SCKM
from livecellx.segment.utils import prep_scs_from_mask_dataset
from livecellx.core.datasets import LiveCellImageDataset
from livecellx.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from livecellx.track.sort_tracker_utils import (
    gen_SORT_detections_input_from_contours,
    update_traj_collection_by_SORT_tracker_detection,
    track_SORT_bbox_from_contours,
    track_SORT_bbox_from_scs,
)
from tests.test_utils import TestHelper


class SingleCellTrajectoryIOTest(TestHelper):
    @classmethod
    def setUpClass(cls):
        # this is a set up method to create a sample SingleCellTrajectory object
        # and any necessary data for the tests
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)

        cls.traj_collection = track_SORT_bbox_from_scs(
            single_cells, dic_dataset, mask_dataset=mask_dataset, max_age=1, min_hits=1
        )

        # extract a single SingleCellTrajectory from the collection and store it in another class attribute
        # here we arbitrarily take the trajectory corresponding to the first track_id we find in the collection
        first_track_id = next(iter(cls.traj_collection.track_id_to_trajectory))
        cls.sct = cls.traj_collection.get_trajectory(first_track_id)

    def setUp(self):
        self.io_out_dir = Path("test_io_output")
        self.io_out_dir.mkdir(exist_ok=True)  # Make sure the directory exists before each test
        self.json_file_path = self.io_out_dir / "test_sc_trajectory.json"

    def test_to_json_dict(self):
        # this test checks if the to_json_dict method works correctly
        result = self.sct.to_json_dict(dataset_json_dir=self.io_out_dir)
        self.assertIsInstance(result, dict)

        # Check if all necessary keys exist and the values are correct
        self.assertEqual(result["track_id"], self.sct.track_id)
        for timeframe, sc in self.sct.timeframe_to_single_cell.items():
            self.assertEqual(
                result["timeframe_to_single_cell"][int(float(timeframe))],
                sc.to_json_dict(dataset_json_dir=self.io_out_dir),
            )

        self.assertDictEqual(result["meta"], self.sct.meta)

        if self.sct.img_dataset is not None:
            self.assertEqual(
                result["meta"]["img_dataset_json_path"],
                str(self.sct.img_dataset.get_default_json_path(out_dir=self.io_out_dir)),
            )
        else:
            self.assertIsNone(result["meta"]["img_dataset_json_path"])

        if self.sct.mask_dataset is not None:
            self.assertEqual(
                result["meta"]["mask_dataset_json_path"],
                str(self.sct.mask_dataset.get_default_json_path(out_dir=self.io_out_dir)),
            )
        else:
            self.assertIsNone(result["meta"]["mask_dataset_json_path"])

    def test_load_from_json_dict(self):
        # this test checks if the load_from_json_dict method works correctly
        # assuming to_json_dict works correctly
        json_dict = self.sct.to_json_dict(dataset_json_dir=self.io_out_dir)

        # define a helper function to write datasets to json files
        def _write_dataset(dataset, json_dir_key, meta_dict):
            if dataset is not None and json_dir_key in meta_dict:
                dir_path = os.path.dirname(meta_dict[json_dir_key])  # get the directory path
                if not os.path.exists(dir_path):  # check if the directory exists
                    os.makedirs(dir_path)  # create the directory
                with open(meta_dict[json_dir_key], "w+") as f:
                    json.dump(dataset.to_json_dict(), f)

        # manually write datasets to their json directories
        _write_dataset(self.sct.img_dataset, "img_dataset_json_path", json_dict["meta"])
        _write_dataset(self.sct.mask_dataset, "mask_dataset_json_path", json_dict["meta"])

        new_sct = SingleCellTrajectory().load_from_json_dict(json_dict)

        self.assertIsInstance(new_sct, SingleCellTrajectory)
        # Check all attributes
        self.assertEqual(new_sct.track_id, self.sct.track_id)

        def _assertTimeframeToSC(new, origin):
            self.assertEqual(new.keys(), origin.keys())
            for key in new.keys():
                new_sc = new[key]
                origin_sc = origin[key]

                self.assertIsInstance(new_sc, SingleCellStatic)
                self.assertIsInstance(origin_sc, SingleCellStatic)

                # Compare the 'bbox' attributes of the SingleCellStatic instances
                np.testing.assert_array_equal(new_sc.bbox, origin_sc.bbox)
                # Compare 'id' attributes
                self.assertEqual(new_sc.id, str(origin_sc.id))
                # Compare 'timeframe' attributes
                self.assertEqual(new_sc.timeframe, origin_sc.timeframe)
                # Compare 'feature_dict' attributes
                self.assertDictEqual(new_sc.feature_dict, origin_sc.feature_dict)
                # Compare 'contour' attributes
                np.testing.assert_array_equal(new_sc.contour, origin_sc.contour)
                # Compare 'meta' attributes
                self.assertDictEqual(new_sc.meta, origin_sc.meta)
                # List of datasets and properties to validate
                datasets = ["img_dataset", "mask_dataset"]
                properties = ["data_dir_path", "ext", "time2url", "name"]

                for dataset in datasets:
                    origin_dataset = getattr(origin_sc, dataset)
                    new_dataset = getattr(new_sc, dataset)

                    if origin_dataset is not None and new_dataset is not None:
                        for prop in properties:
                            origin_prop = getattr(origin_dataset, prop)
                            new_prop = getattr(new_dataset, prop)

                            # Compare as string if property is 'data_dir_path', otherwise compare as is
                            if prop == "data_dir_path":
                                self.assertEqual(str(origin_prop), new_prop)
                            else:
                                self.assertEqual(origin_prop, new_prop)
                # TODO: [smz] to validate dataset_dict
                # Compare 'dataset_dict' attributes
                # self.assertDictEqual(newsc.dataset_dict, originsc.dataset_dict)

        _assertTimeframeToSC(new_sct.timeframe_to_single_cell, self.sct.timeframe_to_single_cell)

        # Check meta dictionary
        self.assertDictEqual(new_sct.meta, self.sct.meta)

        # Define a helper function to compare datasets
        def _test_load_dataset(loaded_dataset, original_dataset):
            if loaded_dataset is None and original_dataset is None:
                return
            elif loaded_dataset is None or original_dataset is None:
                self.fail("One of the datasets is None")
            else:
                self.assertEqual(loaded_dataset.data_dir_path, str(original_dataset.data_dir_path))
                self.assertEqual(loaded_dataset.ext, original_dataset.ext)
                self.assertDictEqual(loaded_dataset.time2url, original_dataset.time2url)
                self.assertEqual(loaded_dataset.name, original_dataset.name)

        # Check equality for img_dataset, mask_dataset.
        # Use the helper function to check the datasets
        _test_load_dataset(new_sct.img_dataset, self.sct.img_dataset)
        _test_load_dataset(new_sct.mask_dataset, self.sct.mask_dataset)

        # Clean up created JSON files
        if json_dict["meta"]["img_dataset_json_path"] is not None and os.path.exists(
            json_dict["meta"]["img_dataset_json_path"]
        ):
            os.remove(json_dict["meta"]["img_dataset_json_path"])
        if json_dict["meta"]["mask_dataset_json_path"] is not None and os.path.exists(
            json_dict["meta"]["mask_dataset_json_path"]
        ):
            os.remove(json_dict["meta"]["mask_dataset_json_path"])

    def test_write_json(self):
        # this test checks if the write_json method works correctly
        self.sct.write_json(path=self.json_file_path, dataset_json_dir=self.io_out_dir)

        with open(self.json_file_path, "r") as f:
            content = json.load(f)

        # Check that the content is a dictionary
        self.assertIsInstance(content, dict)

        # Make a deep copy of the SingleCellTrajectory object and call to_json_dict on the copy.
        sct_copy = copy.deepcopy(self.sct)
        expected_content = sct_copy.to_json_dict(self.io_out_dir)

        # Convert the integer keys to string keys in 'expected_content'
        expected_content["timeframe_to_single_cell"] = {
            str(k): v for k, v in expected_content["timeframe_to_single_cell"].items()
        }

        # Compare the content with the result of to_json_dict using DeepDiff
        diff = DeepDiff(expected_content, content, ignore_order=True)
        self.assertDictEqual(diff, {})

    def test_load_from_json_file(self):
        # Call write_json to write the object to the file
        self.sct.write_json(path=self.json_file_path, dataset_json_dir=self.io_out_dir)

        # Load a new SingleCellTrajectory object from the file
        new_sct = SingleCellTrajectory.load_from_json_file(path=self.json_file_path)

        # Check that the loaded object is a SingleCellTrajectory
        self.assertIsInstance(new_sct, SingleCellTrajectory)

        # Compare the two SingleCellTrajectory objects
        self.assertEqualSct(new_sct, self.sct)

    def tearDown(self):
        # Clean up the test file if it exists
        if self.json_file_path.exists():
            self.json_file_path.unlink()


if __name__ == "__main__":
    unittest.main()
