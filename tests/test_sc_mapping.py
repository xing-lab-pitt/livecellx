import unittest
from unittest.mock import MagicMock
from livecellx.core.sc_mapping import extend_zero_map, SingleCellStatic


class TestExtendZeroMap(unittest.TestCase):
    def test_extend_zero_map(self):
        sample_contour = [[0, 2], [2, 2], [2, 0], [0, 0]]
        non_overlap_contour = [[100, 100], [100, 200], [200, 200], [200, 100]]
        # Create mock SingleCellStatic objects
        sc1 = SingleCellStatic(timeframe=1, img_dataset=None, mask_dataset=None, contour=sample_contour)
        sc2 = SingleCellStatic(timeframe=2, img_dataset=None, mask_dataset=None, contour=non_overlap_contour)
        sc3 = SingleCellStatic(timeframe=3, img_dataset=None, mask_dataset=None, contour=non_overlap_contour)
        sc4 = SingleCellStatic(timeframe=4, img_dataset=None, mask_dataset=None, contour=sample_contour)

        # Create mock scs_by_time dictionary
        scs_by_time = {1: [sc1], 2: [sc2], 3: [sc3], 4: [sc4]}

        # Mock compute_iomin function to always return a value below the threshold
        SingleCellStatic.compute_iomin = MagicMock(return_value=0.3)

        # Call the extend_zero_map function
        result = extend_zero_map(sc1, scs_by_time, threshold=0.5)

        # Check the result
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].timeframe, 1)
        self.assertEqual(result[1].timeframe, 2)
        self.assertEqual(result[2].timeframe, 3)
        self.assertEqual(result[3].timeframe, 4)

    def test_extend_zero_map_case_no_extend(self):
        sample_contour = [[0, 2], [2, 2], [2, 0], [0, 0]]
        non_overlap_contour = [[100, 100], [100, 200], [200, 200], [200, 100]]
        # Create mock SingleCellStatic objects
        sc1 = SingleCellStatic(timeframe=1, img_dataset=None, mask_dataset=None, contour=sample_contour)
        sc2 = SingleCellStatic(timeframe=2, img_dataset=None, mask_dataset=None, contour=non_overlap_contour)
        sc3 = SingleCellStatic(timeframe=3, img_dataset=None, mask_dataset=None, contour=non_overlap_contour)
        sc4 = SingleCellStatic(timeframe=4, img_dataset=None, mask_dataset=None, contour=sample_contour)

        # Create mock scs_by_time dictionary
        scs_by_time = {1: [sc1], 2: [sc2], 3: [sc3], 4: [sc4]}

        # Mock compute_iomin function to always return a value below the threshold
        SingleCellStatic.compute_iomin = MagicMock(return_value=0.6)

        # Call the extend_zero_map function
        result = extend_zero_map(sc1, scs_by_time, threshold=0.5)

        # Check the result
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
