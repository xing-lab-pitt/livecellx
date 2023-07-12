import unittest
import numpy as np
from livecell_tracker.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from livecell_tracker.core.datasets import LiveCellImageDataset


class TestHelper(unittest.TestCase):
    def assertEqualSC(self, sc1, sc2):
        """
        Checks if two instances of SingleCellStatic are equal.

        Parameters:
        - sc1: first instance of SingleCellStatic.
        - sc2: second instance of SingleCellStatic.

        Raises an AssertionError if the instances are not equal.
        """
        self.assertEqual(
            sc1.timeframe,
            sc2.timeframe,
            f"timeframe mismatch: {sc1.timeframe} vs {sc2.timeframe}",
        )
        np.testing.assert_array_equal(
            sc1.bbox,
            sc2.bbox,
            f"bbox mismatch: {sc1.bbox} vs {sc2.bbox}",
        )
        self.assertEqual(
            sc1.feature_dict,
            sc2.feature_dict,
            f"feature_dict mismatch: {sc1.feature_dict} vs {sc2.feature_dict}",
        )
        np.testing.assert_array_equal(
            sc1.contour,
            sc2.contour,
            f"contour mismatch: {sc1.contour} vs {sc2.contour}",
        )
        self.assertEqual(
            str(sc1.id),
            str(sc2.id),
            f"id mismatch: {sc1.id} vs {sc2.id}",
        )
        self.assertEqual(
            sc1.meta,
            sc2.meta,
            f"meta mismatch: {sc1.meta} vs {sc2.meta}",
        )
        self.compare_datasets(sc1.img_dataset, sc2.img_dataset)
        self.compare_datasets(sc1.mask_dataset, sc2.mask_dataset)
        return True

    def assertEqualSCTs(self, sct1: SingleCellTrajectory, sct2: SingleCellTrajectory):
        """
        Checks if two instances of SingleCellTrajectory are equal.

        Parameters:
        - sct1: first instance of SingleCellTrajectory.
        - sct2: second instance of SingleCellTrajectory.

        Raises an AssertionError if the instances are not equal.
        """
        self.assertEqual(sct1.track_id, sct2.track_id, f"track_id mismatch: {sct1.track_id} vs {sct2.track_id}")
        self.assertEqual(
            len(sct1.timeframe_to_single_cell),
            len(sct2.timeframe_to_single_cell),
            "Lengths of timeframe_to_single_cell are different",
        )

        for timeframe, single_cell1 in sct1.timeframe_to_single_cell.items():
            single_cell2 = sct2.timeframe_to_single_cell.get(timeframe)
            self.assertEqualSC(single_cell1, single_cell2)

        self.assertTrue(self.compare_datasets(sct1.img_dataset, sct2.img_dataset), "img_datasets are not the same")
        self.assertTrue(self.compare_datasets(sct1.mask_dataset, sct2.mask_dataset), "mask_datasets are not the same")
        self.assertEqual(
            len(sct1.mother_trajectories), len(sct2.mother_trajectories), "Mother trajectories count mismatch"
        )
        self.assertEqual(
            len(sct1.daughter_trajectories), len(sct2.daughter_trajectories), "Daughter trajectories count mismatch"
        )

        for mother_trajectory1 in sct1.mother_trajectories:
            self.assertTrue(
                any(
                    mother_trajectory1.track_id == mother_trajectory2.track_id
                    for mother_trajectory2 in sct2.mother_trajectories
                ),
                "Mother trajectories do not match",
            )

        for daughter_trajectory1 in sct1.daughter_trajectories:
            self.assertTrue(
                any(
                    daughter_trajectory1.track_id == daughter_trajectory2.track_id
                    for daughter_trajectory2 in sct2.daughter_trajectories
                ),
                "Daughter trajectories do not match",
            )

        self.assertDictEqual(sct1.meta, sct2.meta, "meta are not the same")

    def compare_datasets(self, ds1: LiveCellImageDataset, ds2: LiveCellImageDataset):
        """
        Checks if two instances of LiveCellImageDataset are equal.

        Parameters:
        - ds1: first instance of LiveCellImageDataset.
        - ds2: second instance of LiveCellImageDataset.

        Raises an AssertionError if the instances are not equal.
        """
        if ds1 is None and ds2 is None:
            return True
        elif ds1 is None or ds2 is None:
            self.fail("One of the datasets is None")
        else:
            self.assertEqual(
                str(ds1.data_dir_path),
                str(ds2.data_dir_path),
                f"data_dir_path mismatch: {ds1.data_dir_path} vs {ds2.data_dir_path}",
            )
            self.assertEqual(ds1.ext, ds2.ext, f"ext mismatch: {ds1.ext} vs {ds2.ext}")
            self.assertDictEqual(ds1.time2url, ds2.time2url, f"time2url mismatch: {ds1.time2url} vs {ds2.time2url}")
            self.assertEqual(ds1.name, ds2.name, f"name mismatch: {ds1.name} vs {ds2.name}")
            self.assertEqual(
                ds1.index_by_time,
                ds2.index_by_time,
                f"index_by_time mismatch: {ds1.index_by_time} vs {ds2.index_by_time}",
            )
            self.assertEqual(
                ds1.max_cache_size,
                ds2.max_cache_size,
                f"max_cache_size mismatch: {ds1.max_cache_size} vs {ds2.max_cache_size}",
            )
            return True
