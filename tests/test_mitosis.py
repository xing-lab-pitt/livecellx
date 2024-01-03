import numpy as np
import unittest
from unittest.mock import Mock
from typing import List
from livecellx.trajectory.lca_mitosis_utils import split_mitosis_sample


class TestSplitMitosisSample(unittest.TestCase):
    def setUp(self):
        # Mock SingleCellStatic objects
        self.sc1 = Mock()
        self.sc2 = Mock()
        self.sc3 = Mock()
        self.sc4 = Mock()
        self.sc5 = Mock()

        # Mock get_center method to return specific coordinates
        self.sc1.get_center.return_value = np.array([0, 0])
        self.sc2.get_center.return_value = np.array([1, 1])
        self.sc3.get_center.return_value = np.array([2, 2])
        self.sc4.get_center.return_value = np.array([3, 3])
        self.sc5.get_center.return_value = np.array([3, 3])

        # setup timeframes
        self.sc1.timeframe = 0
        self.sc2.timeframe = 1
        self.sc3.timeframe = 2
        self.sc4.timeframe = 3
        self.sc5.timeframe = 3

    def test_no_mitosis(self):
        # Test case where there is no mitosis event
        sample = [self.sc1, self.sc2]
        result = split_mitosis_sample(sample)
        self.assertEqual(result, [[self.sc1, self.sc2]])

    def test_one_mitosis(self):
        # Test case where there is one mitosis event
        sample = [self.sc1, self.sc2, self.sc3, self.sc4, self.sc5]
        result, break_idx = split_mitosis_sample(sample)
        self.assertEqual(result, [[self.sc1, self.sc2, self.sc3, self.sc4], [self.sc1, self.sc2, self.sc3, self.sc5]])
        self.assertEqual(break_idx, 3)

    def test_no_mitosis(self):
        # Test case where there is no matching trajectory for a cell
        self.sc4.get_center.return_value = np.array([100, 100])  # Set a large distance
        sample = [self.sc1, self.sc2, self.sc3, self.sc4]
        result, break_idx = split_mitosis_sample(sample)
        self.assertEqual(result, [[self.sc1, self.sc2, self.sc3, self.sc4]])


if __name__ == "__main__":
    unittest.main()
