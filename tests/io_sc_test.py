import json
import unittest
from pathlib import Path
import numpy as np
from livecell_tracker import sample_data
from livecell_tracker.core.sc_key_manager import SingleCellMetaKeyManager as SCKM
from livecell_tracker.segment.utils import prep_scs_from_mask_dataset
from livecell_tracker.core.datasets import LiveCellImageDataset
from livecell_tracker.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from tests.test_utils import TestHelper


class SingleCellStaticIOTest(TestHelper):
    @classmethod
    def setUpClass(cls):
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        cls.cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)
        cls.cell = cls.cells[0]
        cls.io_out_dir = None
        cls.img_dataset = None
        cls.mask_dataset = None

    def setUp(self):
        self.io_out_dir = Path("test_io_output")
        self.io_out_dir.mkdir(exist_ok=True)  # Make sure the directory exists before each test

    def validate_properties(self, cell, sc_json_dict):
        self.assertEqual(str(cell.id), sc_json_dict["id"], "id does not match")
        self.assertEqual(cell.timeframe, sc_json_dict["timeframe"], "timeframe does not match")
        np.testing.assert_array_equal(cell.bbox, np.array(sc_json_dict["bbox"]), "bbox does not match")
        self.assertEqual(cell.feature_dict, sc_json_dict["feature_dict"], "feature_dict does not match")
        np.testing.assert_array_equal(cell.contour, np.array(sc_json_dict["contour"]), "contour does not match")
        # Load the meta from JSON string to dict before comparing
        sc_meta = sc_json_dict["meta"]
        self.assertEqual(cell.meta, sc_meta, "meta does not match")

    def test_to_json_dict(self):
        for include_dataset_json in [True, False]:
            for has_dataset_json_dir in [True, False]:
                dataset_json_dir = self.io_out_dir if has_dataset_json_dir else None
                result = self.cell.to_json_dict(
                    include_dataset_json=include_dataset_json, dataset_json_dir=dataset_json_dir
                )

                assert isinstance(result, dict)
                assert result["timeframe"] == self.cell.timeframe
                assert result["bbox"] == self.cell.bbox.tolist()
                assert result["feature_dict"] == self.cell.feature_dict
                assert result["contour"] == self.cell.contour.tolist()
                assert result["id"] == str(self.cell.id)
                assert result["meta"] == self.cell.meta

                # Check the 'dataset_json' field
                if include_dataset_json:
                    assert "dataset_json" in result
                    dataset_json = result["dataset_json"]
                    assert isinstance(dataset_json, dict)
                    assert "name" in dataset_json
                    assert "data_dir_path" in dataset_json
                    assert "max_cache_size" in dataset_json
                    assert "ext" in dataset_json
                    assert "time2url" in dataset_json
                else:
                    assert "dataset_json" not in result

                # Check the 'dataset_json_dir' field
                if dataset_json_dir:
                    assert "dataset_json_dir" in result
                    assert result["dataset_json_dir"] == str(dataset_json_dir)
                else:
                    assert "dataset_json_dir" not in result

                # Check paths and dataset json files for img_dataset and mask_dataset
                if self.cell.img_dataset is not None:
                    assert SCKM.JSON_IMG_DATASET_PATH in result["meta"]
                    assert result["meta"][SCKM.JSON_IMG_DATASET_PATH] == str(
                        self.cell.img_dataset.get_default_json_path(out_dir=dataset_json_dir)
                    )
                else:
                    assert SCKM.JSON_IMG_DATASET_PATH not in result["meta"]

                if self.cell.mask_dataset is not None:
                    assert SCKM.JSON_MASK_DATASET_PATH in result["meta"]
                    assert result["meta"][SCKM.JSON_MASK_DATASET_PATH] == str(
                        self.cell.mask_dataset.get_default_json_path(out_dir=dataset_json_dir)
                    )
                else:
                    assert SCKM.JSON_MASK_DATASET_PATH not in result["meta"]

    def test_load_from_json_dict(self):
        json_dict = self.cell.to_json_dict(dataset_json_dir=self.io_out_dir)

        new_cell = SingleCellStatic().load_from_json_dict(json_dict)

        # Now validate the properties using ssertEqualSC method
        self.assertEqualSc(self.cell, new_cell)

    def test_write_single_cells_json(self):
        json_path = self.io_out_dir / "test_single_cells.json"

        SingleCellStatic.write_single_cells_json(self.cells, str(json_path), str(self.io_out_dir))
        self.assertTrue(json_path.is_file())
        # Now check that the file was correctly written
        with open(json_path, "r") as f:
            sc_json_dict_list = json.load(f)

        # Check that the json data matches the cell data
        for i, cell in enumerate(self.cells):
            self.validate_properties(cell, sc_json_dict_list[i])

    def test_load_single_cells_json(self):
        json_path = self.io_out_dir / "test_single_cells.json"

        SingleCellStatic.write_single_cells_json(self.cells, str(json_path), str(self.io_out_dir))
        loaded_cells = SingleCellStatic.load_single_cells_json(str(json_path))

        # Check that the loaded cells match the original ones
        for i, loaded_cell in enumerate(loaded_cells):
            original_cell = self.cells[i]
            self.assertEqualSc(original_cell, loaded_cell)

    def test_write_json(self):
        # Test write to file
        json_path = self.io_out_dir / "test_single_cell.json"
        self.cell.write_json(str(json_path))
        self.assertTrue(json_path.is_file(), "JSON file not created")
        # Now check that the file was correctly written
        with open(json_path, "r") as f:
            sc_json_dict = json.load(f)
        # Validate the properties
        self.validate_properties(self.cell, sc_json_dict)

        # Test write to string
        json_str = self.cell.write_json()
        sc_json_dict = json.loads(json_str)
        # Validate the properties
        self.validate_properties(self.cell, sc_json_dict)

    def tearDown(self):
        # This method will be called after each test. Clean up the test fixture here.
        # If setup included opening files or establishing network connections, close them here.
        json_path = self.io_out_dir / "test_single_cells.json"
        if json_path.is_file():
            json_path.unlink()


if __name__ == "__main__":
    unittest.main()
