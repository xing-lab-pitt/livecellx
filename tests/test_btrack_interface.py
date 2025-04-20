"""
Unit tests for the btrack interface in livecellx.
"""

import unittest
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from livecellx import sample_data
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.core.single_cell import SingleCellTrajectoryCollection
from livecellx.trajectory.feature_extractors import compute_skimage_regionprops, parallelize_compute_features
from livecellx.preprocess.utils import normalize_img_to_uint8
from livecellx.track.btrack_tracker_utils import track_btrack_from_scs


class TestBtrackInterface(unittest.TestCase):
    """Test case for the btrack interface in livecellx."""

    def setUp(self):
        """Set up test data."""
        # Load sample data
        self.dic_dataset, self.mask_dataset = sample_data.tutorial_three_image_sys()

        # Prepare single cell objects
        self.single_cells = prep_scs_from_mask_dataset(self.mask_dataset, self.dic_dataset)

        # Compute features for the single cells using the feature_extractors module
        _, self.single_cells = parallelize_compute_features(
            self.single_cells,
            compute_skimage_regionprops,
            params={
                "feature_key": "skimage",
                "preprocess_img_func": normalize_img_to_uint8,
                "sc_level_normalize": True,
            },
            replace_feature=True,
            verbose=True
        )

        # Store original IDs for verification
        self.original_ids = {sc.id for sc in self.single_cells}

    def test_track_btrack_from_scs_interface(self):
        """Test the track_btrack_from_scs interface."""
        # Track the single cells using the btrack interface
        trajectories = track_btrack_from_scs(
            self.single_cells,
            raw_imgs=self.dic_dataset,
            mask_dataset=self.mask_dataset,
            feature_names=['area', 'perimeter', 'eccentricity'],
            max_search_radius=20.0
        )

        # Check that the result is a SingleCellTrajectoryCollection
        self.assertIsInstance(trajectories, SingleCellTrajectoryCollection)

        # Check that trajectories were created
        self.assertGreater(len(trajectories.track_id_to_trajectory), 0)

        # Check that each trajectory has cells
        for track_id, trajectory in trajectories.track_id_to_trajectory.items():
            self.assertGreater(len(trajectory.timeframe_to_single_cell), 0)

            # Check that each cell in the trajectory has the original ID restored
            for timeframe, sc in trajectory.timeframe_to_single_cell.items():
                self.assertIn(sc.id, self.original_ids)

                # Check that the btrack ID is stored in sc.uns
                self.assertTrue(hasattr(sc, 'uns'))
                self.assertIn('btrack_id', sc.uns)
                self.assertEqual(sc.uns['btrack_id'], track_id)

        # Visualize the trajectories
        plt.figure(figsize=(10, 8))

        # Plot the trajectories
        for track_id, trajectory in trajectories.track_id_to_trajectory.items():
            # Get the positions of the cells in the trajectory
            positions = []
            for timeframe in sorted(trajectory.timeframe_to_single_cell.keys()):
                sc = trajectory.timeframe_to_single_cell[timeframe]
                x = (sc.bbox[0] + sc.bbox[2]) / 2
                y = (sc.bbox[1] + sc.bbox[3]) / 2
                positions.append((x, y))

            # Plot the trajectory
            if positions:
                x_vals, y_vals = zip(*positions)
                plt.plot(x_vals, y_vals, '-o', label=f'Track {track_id}')

        plt.title('Cell Trajectories')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)

        # Save the figure
        plt.savefig('btrack_interface_trajectories.png')
        plt.close()

        # Check that the figure was created
        self.assertTrue(os.path.exists('btrack_interface_trajectories.png'))

    def test_track_btrack_from_scs_with_dataframe(self):
        """Test the track_btrack_from_scs interface with DataFrame output."""
        # Track the single cells using the btrack interface with DataFrame output
        trajectories, df = track_btrack_from_scs(
            self.single_cells,
            raw_imgs=self.dic_dataset,
            mask_dataset=self.mask_dataset,
            feature_names=['area', 'perimeter', 'eccentricity'],
            max_search_radius=20.0,
            return_dataframe=True
        )

        # Check that the result is a SingleCellTrajectoryCollection and a DataFrame
        self.assertIsInstance(trajectories, SingleCellTrajectoryCollection)
        self.assertIsInstance(df, pd.DataFrame)

        # Check that the DataFrame has the correct columns
        self.assertIn('track_id', df.columns)
        self.assertIn('frame', df.columns)
        self.assertIn('x', df.columns)
        self.assertIn('y', df.columns)
        self.assertIn('z', df.columns)

        # Check that the DataFrame has the correct number of rows
        self.assertGreater(len(df), 0)

        # Check that the original IDs are included in the DataFrame
        self.assertIn('original_id', df.columns)

        # Check that each original ID in the DataFrame is one of the original IDs
        for _, row in df.iterrows():
            self.assertIn(row['original_id'], self.original_ids)

    def test_track_btrack_with_string_ids(self):
        """Test tracking with string IDs."""
        # Assign string IDs to the single cells
        for i, sc in enumerate(self.single_cells):
            sc.id = f"cell_{i}"

        # Track the single cells using the btrack interface
        trajectories = track_btrack_from_scs(
            self.single_cells,
            raw_imgs=self.dic_dataset,
            mask_dataset=self.mask_dataset,
            feature_names=['area', 'perimeter', 'eccentricity'],
            max_search_radius=20.0
        )

        # Check that the result is a SingleCellTrajectoryCollection
        self.assertIsInstance(trajectories, SingleCellTrajectoryCollection)

        # Check that trajectories were created
        self.assertGreater(len(trajectories.track_id_to_trajectory), 0)

        # Check that each trajectory has cells
        for track_id, trajectory in trajectories.track_id_to_trajectory.items():
            self.assertGreater(len(trajectory.timeframe_to_single_cell), 0)

            # Check that each cell in the trajectory has a string ID
            for timeframe, sc in trajectory.timeframe_to_single_cell.items():
                self.assertIsInstance(sc.id, str)
                self.assertTrue(sc.id.startswith("cell_"))

                # Check that the btrack ID is stored in sc.uns
                self.assertTrue(hasattr(sc, 'uns'))
                self.assertIn('btrack_id', sc.uns)
                self.assertEqual(sc.uns['btrack_id'], track_id)

    def test_track_btrack_with_uuid_ids(self):
        """Test tracking with UUID IDs."""
        # Import UUID
        import uuid

        # Assign UUID IDs to the single cells
        for i, sc in enumerate(self.single_cells):
            sc.id = uuid.uuid4()

        # Track the single cells using the btrack interface
        trajectories = track_btrack_from_scs(
            self.single_cells,
            raw_imgs=self.dic_dataset,
            mask_dataset=self.mask_dataset,
            feature_names=['area', 'perimeter', 'eccentricity'],
            max_search_radius=20.0
        )

        # Check that the result is a SingleCellTrajectoryCollection
        self.assertIsInstance(trajectories, SingleCellTrajectoryCollection)

        # Check that trajectories were created
        self.assertGreater(len(trajectories.track_id_to_trajectory), 0)

        # Check that each trajectory has cells
        for track_id, trajectory in trajectories.track_id_to_trajectory.items():
            self.assertGreater(len(trajectory.timeframe_to_single_cell), 0)

            # Check that each cell in the trajectory has a UUID ID
            for timeframe, sc in trajectory.timeframe_to_single_cell.items():
                self.assertIsInstance(sc.id, uuid.UUID)

                # Check that the btrack ID is stored in sc.uns
                self.assertTrue(hasattr(sc, 'uns'))
                self.assertIn('btrack_id', sc.uns)
                self.assertEqual(sc.uns['btrack_id'], track_id)


if __name__ == '__main__':
    unittest.main()
