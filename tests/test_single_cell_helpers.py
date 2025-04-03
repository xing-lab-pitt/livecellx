import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from livecellx import sample_data
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.core.single_cell import (
    SingleCellStatic,
    SingleCellTrajectory,
    SingleCellTrajectoryCollection,
    show_sct_on_grid,
    sample_samples_from_sctc,
    create_sctc_from_scs,
    filter_sctc_by_time_span,
)
from livecellx.track.sort_tracker_utils import track_SORT_bbox_from_scs
from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset
from tests.test_utils import TestHelper


class SingleCellHelpersTestCase(TestHelper):
    @classmethod
    def setUpClass(cls):
        # Load sample data
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        cls.single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)
        cls.img_dataset = dic_dataset
        cls.mask_dataset = mask_dataset

        # Create a trajectory collection
        cls.traj_collection = track_SORT_bbox_from_scs(
            cls.single_cells, dic_dataset, mask_dataset=mask_dataset, max_age=1, min_hits=1
        )

    def setUp(self):
        # Get the first trajectory for testing
        self.trajectory = next(iter(self.traj_collection.track_id_to_trajectory.values()))

    def test_show_sct_on_grid(self):
        """Test showing a SingleCellTrajectory on a grid"""
        # Show the trajectory on a grid
        fig, axes = show_sct_on_grid(self.trajectory, nr=2, nc=2, start=0, interval=1, padding=10, verbose=True)

        # Check that the figure and axes are not None
        self.assertIsNotNone(fig)
        self.assertIsNotNone(axes)

        # Close the figure
        plt.close(fig)

    def test_sample_samples_from_sctc(self):
        """Test sampling samples from a SingleCellTrajectoryCollection"""
        # Add src_dir to meta for each single cell
        for track_id, trajectory in self.traj_collection:
            for _, sc in trajectory:
                sc.meta["src_dir"] = "test_dir"

        # Sample samples from the trajectory collection
        samples, extra_info = sample_samples_from_sctc(
            self.traj_collection, objective_sample_num=5, seed=0, length_range=(2, 3)
        )

        # Check that samples were returned
        self.assertIsNotNone(samples)
        self.assertIsNotNone(extra_info)

        # Check that the number of samples is less than or equal to the objective
        self.assertLessEqual(len(samples), 5)

        # Check that each sample has the expected length
        for sample in samples:
            self.assertGreaterEqual(len(sample), 2)
            self.assertLessEqual(len(sample), 3)

    def test_create_sctc_from_scs(self):
        """Test creating a SingleCellTrajectoryCollection from SingleCellStatic instances"""
        # Create a trajectory collection from single cells
        sctc = create_sctc_from_scs(self.single_cells)

        # Check that the trajectory collection is not None
        self.assertIsNotNone(sctc)

        # Check that the trajectory collection has the expected number of trajectories
        self.assertEqual(len(sctc), len(self.single_cells))

        # Check that each trajectory has exactly one cell
        for _, sct in sctc:
            self.assertEqual(len(sct), 1)

    def test_filter_sctc_by_time_span(self):
        """Test filtering a SingleCellTrajectoryCollection by time span"""
        # Create a trajectory collection with cells at different timeframes
        sctc = SingleCellTrajectoryCollection()

        # Add trajectories with different time spans
        for i in range(5):
            sct = SingleCellTrajectory(track_id=i)
            for j in range(i, i + 3):  # Trajectory i spans from i to i+2
                sc = self.single_cells[0].copy()
                sc.timeframe = j
                sct.add_sc(sc)
            sctc.add_trajectory(sct)

        # Filter the trajectory collection to include only trajectories with a time span of at least 3
        filtered_sctc = filter_sctc_by_time_span(sctc, time_span=(0, 2))

        # Check that the filtered collection has the expected number of trajectories
        self.assertEqual(
            len(filtered_sctc), 3
        )  # Trajectories with track_id 0, 1, 2 have timeframes that overlap with (0, 2)

        # Filter the trajectory collection to include only trajectories with a time span of at least 4
        filtered_sctc = filter_sctc_by_time_span(sctc, time_span=(10, 20))

        # Check that the filtered collection has the expected number of trajectories
        self.assertEqual(len(filtered_sctc), 0)  # No trajectories have timeframes that overlap with (10, 20)


if __name__ == "__main__":
    unittest.main()
