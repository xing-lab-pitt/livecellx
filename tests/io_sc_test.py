import json
import unittest
from pathlib import Path
import numpy as np
from livecell_tracker import sample_data
from livecell_tracker.core.sc_key_manager import SingleCellMetaKeyManager as SCKM
from livecell_tracker.livecell_logger import main_warning
from livecell_tracker.segment.utils import prep_scs_from_mask_dataset
from livecell_tracker.core.datasets import LiveCellImageDataset
from livecell_tracker.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)


class SingleCellStaticIOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        cls.cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)
        cls.cell = cls.cells[0]
        cls.include_dataset_json = False
        cls.dataset_json_dir = None
        cls.img_dataset = None
        cls.mask_dataset = None

    def setUp(self):
        self.io_out_dir = Path("test_io_output")
        self.io_out_dir.mkdir(exist_ok=True)  # Make sure the directory exists before each test

    # TODO
    def test_read_traj_collection(self):
        return
        traj_collection_json_path = "../datasets/test_data/traj_analysis/test_trajs.json"
        traj_collection_json = json.load(open(traj_collection_json_path, "r"))
        trajectory_collection = SingleCellTrajectoryCollection().load_from_json_dict(traj_collection_json)

        # TODO: recursively check all the trajectories and all single cell objects

    def test_to_json_dict(self):
        result = self.cell.to_json_dict(include_dataset_json=True, dataset_json_dir=self.dataset_json_dir)

        assert isinstance(result, dict)
        assert result["timeframe"] == self.cell.timeframe
        assert result["bbox"] == self.cell.bbox.tolist()
        assert result["feature_dict"] == self.cell.feature_dict
        assert result["contour"] == self.cell.contour.tolist()
        assert result["id"] == str(self.cell.id)
        assert result["meta"] == self.cell.meta_copy

        if self.include_dataset_json:
            assert "dataset_json" in result
            dataset_json = result["dataset_json"]
            assert isinstance(dataset_json, dict)
            assert "name" in dataset_json
            assert "data_dir_path" in dataset_json
            assert "max_cache_size" in dataset_json
            assert "ext" in dataset_json
            assert "time2url" in dataset_json

        if self.dataset_json_dir:
            assert "dataset_json_dir" in result
            assert result["dataset_json_dir"] == str(self.dataset_json_dir)
            assert SCKM.JSON_IMG_DATASET_JSON_PATH in result
            assert result[SCKM.JSON_IMG_DATASET_JSON_PATH] == str(
                self.cell.img_dataset.get_default_json_path(out_dir=self.dataset_json_dir)
            )
            assert SCKM.JSON_MASK_DATASET_JSON_PATH in result
            assert result[SCKM.JSON_MASK_DATASET_JSON_PATH] == str(
                self.cell.mask_dataset.get_default_json_path(out_dir=self.dataset_json_dir)
            )

    def test_load_from_json_dict(self):
        json_dict = self.cell.to_json_dict()

        new_cell = SingleCellStatic()
        new_cell.load_from_json_dict(json_dict)

        # Now validate the properties
        # Validate timeframe
        self.assertEqual(self.cell.timeframe, new_cell.timeframe, "timeframe does not match")
        # Validate bbox
        np.testing.assert_array_equal(self.cell.bbox, new_cell.bbox, "bbox does not match")
        # Validate feature_dict
        self.assertEqual(self.cell.feature_dict, new_cell.feature_dict, "feature_dict does not match")
        # Validate contour
        np.testing.assert_array_equal(self.cell.contour, new_cell.contour, "contour does not match")
        # Validate id
        self.assertEqual(str(self.cell.id), new_cell.id, "id does not match")
        # Validate meta
        self.assertEqual(self.cell.meta, new_cell.meta, "meta does not match")

        # Validate img_dataset and mask_dataset
        for prop in ["data_dir_path", "ext", "time2url", "name"]:
            if self.cell.img_dataset is not None and new_cell.img_dataset is not None:
                self.assertEqual(getattr(self.cell.img_dataset, prop), getattr(new_cell.img_dataset, prop))

            if self.cell.mask_dataset is not None and new_cell.mask_dataset is not None:
                self.assertEqual(getattr(self.cell.mask_dataset, prop), getattr(new_cell.mask_dataset, prop))

    def test_write_single_cells_json(self):
        json_path = self.io_out_dir / "test_single_cells.json"

        SingleCellStatic.write_single_cells_json(self.cells, str(json_path), str(self.io_out_dir))
        self.assertTrue(json_path.is_file())
        # Now check that the file was correctly written
        with open(json_path, "r") as f:
            sc_json_dict_list = json.load(f)

        # Check that the json data matches the cell data
        for i, cell in enumerate(self.cells):
            self.assertEqual(str(cell.id), sc_json_dict_list[i]["id"], "id does not match")
            self.assertEqual(cell.timeframe, sc_json_dict_list[i]["timeframe"], "timeframe does not match")
            np.testing.assert_array_equal(cell.bbox, np.array(sc_json_dict_list[i]["bbox"]), "bbox does not match")
            self.assertEqual(cell.feature_dict, sc_json_dict_list[i]["feature_dict"], "feature_dict does not match")
            np.testing.assert_array_equal(
                cell.contour, np.array(sc_json_dict_list[i]["contour"]), "contour does not match"
            )
            self.assertEqual(cell.meta, sc_json_dict_list[i]["meta"], "meta does not match")

    def test_load_single_cells_json(self):
        json_path = self.io_out_dir / "test_single_cells.json"

        SingleCellStatic.write_single_cells_json(self.cells, str(json_path), str(self.io_out_dir))
        loaded_cells = SingleCellStatic.load_single_cells_json(str(json_path))

        # Check that the loaded cells match the original ones
        for i, loaded_cell in enumerate(loaded_cells):
            original_cell = self.cells[i]
            # Validate timeframe
            self.assertEqual(original_cell.timeframe, loaded_cell.timeframe, "timeframe does not match")
            # Validate bbox
            np.testing.assert_array_equal(original_cell.bbox, loaded_cell.bbox, "bbox does not match")
            # Validate feature_dict
            self.assertEqual(original_cell.feature_dict, loaded_cell.feature_dict, "feature_dict does not match")
            # Validate contour
            np.testing.assert_array_equal(original_cell.contour, loaded_cell.contour, "contour does not match")
            # Validate id
            self.assertEqual(str(original_cell.id), loaded_cell.id, "id does not match")
            # Validate meta
            self.assertEqual(original_cell.meta, loaded_cell.meta, "meta does not match")

    def test_write_json(self):
        # Test write to file
        json_path = self.io_out_dir / "test_single_cell.json"
        self.cell.write_json(str(json_path))
        self.assertTrue(json_path.is_file(), "JSON file not created")
        # Now check that the file was correctly written
        with open(json_path, "r") as f:
            sc_json_dict = json.load(f)
        # Validate the properties
        self.assertEqual(str(self.cell.id), sc_json_dict["id"], "id does not match")
        self.assertEqual(self.cell.timeframe, sc_json_dict["timeframe"], "timeframe does not match")
        np.testing.assert_array_equal(self.cell.bbox, np.array(sc_json_dict["bbox"]), "bbox does not match")
        self.assertEqual(self.cell.feature_dict, sc_json_dict["feature_dict"], "feature_dict does not match")
        np.testing.assert_array_equal(self.cell.contour, np.array(sc_json_dict["contour"]), "contour does not match")
        self.assertEqual(self.cell.meta, sc_json_dict["meta"], "meta does not match")

        # Test write to string
        json_str = self.cell.write_json()
        sc_json_dict = json.loads(json_str)
        # Validate the properties
        self.assertEqual(str(self.cell.id), sc_json_dict["id"], "id does not match")
        self.assertEqual(self.cell.timeframe, sc_json_dict["timeframe"], "timeframe does not match")
        np.testing.assert_array_equal(self.cell.bbox, np.array(sc_json_dict["bbox"]), "bbox does not match")
        self.assertEqual(self.cell.feature_dict, sc_json_dict["feature_dict"], "feature_dict does not match")
        np.testing.assert_array_equal(self.cell.contour, np.array(sc_json_dict["contour"]), "contour does not match")
        self.assertEqual(self.cell.meta, sc_json_dict["meta"], "meta does not match")

    def tearDown(self):
        # This method will be called after each test. Clean up the test fixture here.
        # If setup included opening files or establishing network connections, close them here.
        json_path = self.io_out_dir / "test_single_cells.json"
        if json_path.is_file():
            json_path.unlink()


if __name__ == "__main__":
    unittest.main()
