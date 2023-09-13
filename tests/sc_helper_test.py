import unittest

from livecellx import sample_data
from livecellx.core.single_cell import create_sctc_from_scs, filter_sctc_by_time_span
from livecellx.segment.utils import prep_scs_from_mask_dataset
from livecellx.core import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection
from livecellx.track.sort_tracker_utils import (
    gen_SORT_detections_input_from_contours,
    update_traj_collection_by_SORT_tracker_detection,
    track_SORT_bbox_from_contours,
    track_SORT_bbox_from_scs,
)


class TestScUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # this is a set up method to create a sample SingleCellTrajectory object
        # and any necessary data for the tests
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        cls.single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)
        cls.traj_collection = track_SORT_bbox_from_scs(
            cls.single_cells, dic_dataset, mask_dataset=mask_dataset, max_age=1, min_hits=1
        )

    def test_create_sctc_from_scs(self):
        # Create some SingleCellStatic objects for testing
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        self.single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)

        sctc = create_sctc_from_scs(self.single_cells)

        # Test that output is a SingleCellTrajectoryCollection
        self.assertIsInstance(sctc, SingleCellTrajectoryCollection)

        # Test that each SingleCellTrajectory in the collection has exactly one cell
        for _, sct in sctc:
            self.assertEqual(len(sct), 1)

        # Test that each cell from the input single_cells is in the SingleCellTrajectoryCollection
        cells_in_sctc = {cell for _, sct in sctc for cell in sct.timeframe_to_single_cell.values()}
        self.assertSetEqual(set(self.single_cells), cells_in_sctc)

    def test_filter_sctc_by_time_span(self):
        new_sctc = filter_sctc_by_time_span(self.traj_collection, time_span=(0, 1))

        # Test that output is a SingleCellTrajectoryCollection
        self.assertIsInstance(new_sctc, SingleCellTrajectoryCollection)

        # Test that the new collection only contains trajectories within the given time span
        for _, sct in new_sctc:
            for timeframe in sct.timeframe_set:
                self.assertTrue(0 <= timeframe <= 1, f"Timeframe {timeframe} is outside of the span (0, 1)")


if __name__ == "__main__":
    unittest.main()
