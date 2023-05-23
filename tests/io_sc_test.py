import json
import unittest
import glob
from pathlib import Path
import numpy as np
from livecell_tracker import sample_data
from livecell_tracker.segment.utils import prep_scs_from_mask_dataset
from livecell_tracker.core.datasets import LiveCellImageDataset
from livecell_tracker.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
class SingleCellStaticIOTest(unittest.TestCase): 
    def setUp(self):
        io_out_dir = Path("test_io_output")
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)

        self.cell = single_cells[0]
        self.include_dataset_json = False
        self.dataset_json_dir = None
        self.img_dataset = None
        self.mask_dataset = None


    def tearDown(self):
        # This method will be called after each test. Clean up the test fixture here.
        # If setup included opening files or establishing network connections, close them here.
    
        pass

    # TODO
    def test_read_traj_collection(self):
        return
        traj_collection_json_path = "../datasets/test_data/traj_analysis/test_trajs.json"
        traj_collection_json = json.load(open(traj_collection_json_path, "r"))
        trajectory_collection = SingleCellTrajectoryCollection().load_from_json_dict(traj_collection_json)

        # TODO: recursively check all the trajectories and all single cell objects

    def test_to_json_dict(self):
        result = self.cell.to_json_dict()

        assert isinstance(result, dict)
        assert result['timeframe'] == self.cell.timeframe
        assert result['bbox'] == self.cell.bbox.tolist()
        assert result['feature_dict'] == self.cell.feature_dict
        assert result['contour'] == self.cell.contour.tolist()
        assert result['meta'] == self.cell.meta_copy
        assert result['id'] == str(self.cell.id)
        if self.include_dataset_json:
            assert result['id'] == str(self.cell.id)
        if self.dataset_json_dir:
            assert result['id'] == str(self.cell.id)

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
        self.assertEqual(self.cell.id, new_cell.id, "id does not match")
        # Validate meta
        self.assertEqual(self.cell.meta, new_cell.meta, "meta does not match")

if __name__ == '__main__':
    unittest.main()