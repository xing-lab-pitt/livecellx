import json
import unittest
from pathlib import Path
import numpy as np
from livecellx import sample_data
from livecellx.core.sc_key_manager import SingleCellMetaKeyManager as SCKM
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.core.datasets import LiveCellImageDataset
from livecellx.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from livecellx.core.single_cell import Organelle
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
        for cell in cls.cells:
            cell.uns["_test_uns_key"] = "_test_uns_value"

    def setUp(self):
        self.io_out_dir = Path("test_io_output")
        self.io_out_dir.mkdir(exist_ok=True)  # Make sure the directory exists before each test

    def test_organelle_io(self):
        import numpy as np
        dummy_contour = np.array([[10, 10], [10, 20], [20, 20], [20, 10]])
        sub = Organelle(
            organelle_type="nucleolus",
            timeframe=0,
            contour=dummy_contour,
            bbox=[10, 10, 21, 21],
            empty_cell=True,
        )
        sub_json = sub.to_json_dict()
        new_sub = Organelle(organelle_type="").load_from_json_dict(sub_json)
        assert new_sub.organelle_type == "nucleolus"
        assert new_sub.contour.shape == (4, 2)
        assert list(new_sub.bbox) == [10, 10, 21, 21]
        print("Organelle IO test passed.")

    def tearDown(self):
        # This method will be called after each test. Clean up the test fixture here.
        # If setup included opening files or establishing network connections, close them here.
        json_path = self.io_out_dir / "test_single_cells.json"
        if json_path.is_file():
            json_path.unlink()


if __name__ == "__main__":
    unittest.main()

