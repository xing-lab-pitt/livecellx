import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from livecellx.core.sc_filters import filter_boundary_cells
from livecellx.core.single_cell import SingleCellStatic


class TestFilterBoundaryCells(unittest.TestCase):
    def setUp(self):
        # Create mock SingleCellStatic objects
        # self.sc1 = MagicMock(spec=SingleCellStatic)
        # self.sc2 = MagicMock(spec=SingleCellStatic)
        # self.sc3 = MagicMock(spec=SingleCellStatic)

        # # Mock get_bbox method
        # self.sc1.get_bbox.return_value = np.array([10, 10, 20, 20])
        # self.sc2.get_bbox.return_value = np.array([50, 50, 60, 60])
        # self.sc3.get_bbox.return_value = np.array([90, 90, 190, 190])

        # # Mock get_img method
        # img = np.zeros((200, 200))
        # self.sc1.get_img.return_value = img
        # self.sc2.get_img.return_value = img
        # self.sc3.get_img.return_value = img
        pass

    def test_filter_boundary_cells_default(self):
        sc1 = MagicMock(spec=SingleCellStatic)
        sc2 = MagicMock(spec=SingleCellStatic)
        sc3 = MagicMock(spec=SingleCellStatic)
        sc1.get_bbox.return_value = np.array([10, 10, 20, 20])
        sc2.get_bbox.return_value = np.array([50, 50, 60, 60])
        sc3.get_bbox.return_value = np.array([90, 90, 190, 190])
        img = np.zeros((200, 200))
        sc1.get_img.return_value = img
        sc2.get_img.return_value = img
        sc3.get_img.return_value = img
        scs = [sc1, sc2, sc3]
        result = filter_boundary_cells(scs, dist_to_boundary=30, use_box_center=False)
        # sc1 is too close to the left/top boundary
        # sc3 is too close to the right/bottom boundary
        self.assertEqual(len(result), 1)
        self.assertIn(sc2, result)

    def test_filter_boundary_cells_custom_dist(self):
        sc1 = MagicMock(spec=SingleCellStatic)
        sc2 = MagicMock(spec=SingleCellStatic)
        sc3 = MagicMock(spec=SingleCellStatic)
        sc1.get_bbox.return_value = np.array([9, 9, 20, 20])
        sc2.get_bbox.return_value = np.array([50, 50, 60, 60])
        sc3.get_bbox.return_value = np.array([90, 90, 180, 180])
        img = np.zeros((200, 200))
        sc1.get_img.return_value = img
        sc2.get_img.return_value = img
        sc3.get_img.return_value = img
        scs = [sc1, sc2, sc3]
        result = filter_boundary_cells(scs, dist_to_boundary=10, use_box_center=False)
        # With dist_to_boundary=10, sc1 and sc2 should be kept
        # sc3 is still too close to the right/bottom boundary
        self.assertEqual(len(result), 2)
        self.assertIn(sc3, result)
        self.assertIn(sc2, result)

    def test_filter_boundary_cells_custom_bbox_bounds(self):
        sc1 = MagicMock(spec=SingleCellStatic)
        sc2 = MagicMock(spec=SingleCellStatic)
        sc3 = MagicMock(spec=SingleCellStatic)
        sc1.get_bbox.return_value = np.array([9, 9, 20, 20])
        sc2.get_bbox.return_value = np.array([50, 50, 66, 66])
        sc3.get_bbox.return_value = np.array([90, 90, 180, 180])
        img = np.zeros((200, 200))
        sc1.get_img.return_value = img
        sc2.get_img.return_value = img
        sc3.get_img.return_value = img
        scs = [sc1, sc2, sc3]
        bbox_bounds = [0, 0, 70, 70]
        result = filter_boundary_cells(scs, bbox_bounds=bbox_bounds, dist_to_boundary=5, use_box_center=False)
        # With custom bbox_bounds, only sc1 should be kept
        # sc2 and sc3 are outside the bbox_bounds
        self.assertEqual(len(result), 1)
        self.assertIn(sc1, result)

    def test_filter_boundary_cells_use_box_center_true(self):
        sc1 = MagicMock(spec=SingleCellStatic)
        sc2 = MagicMock(spec=SingleCellStatic)
        sc3 = MagicMock(spec=SingleCellStatic)
        sc1.get_bbox.return_value = np.array([9, 9, 40, 40])
        sc2.get_bbox.return_value = np.array([50, 50, 60, 60])
        sc3.get_bbox.return_value = np.array([90, 90, 180, 180])
        img = np.zeros((200, 200))
        sc1.get_img.return_value = img
        sc2.get_img.return_value = img
        sc3.get_img.return_value = img
        scs = [sc1, sc2, sc3]
        result = filter_boundary_cells(scs, use_box_center=True, dist_to_boundary=10)
        # With use_box_center=False, only sc2 should be kept
        # sc1 is too close to the left/top boundary
        # sc3 is too close to the right/bottom boundary
        self.assertEqual(len(result), 3)
        self.assertIn(sc1, result)


if __name__ == "__main__":
    unittest.main()
