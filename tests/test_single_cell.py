import unittest
from unittest.mock import MagicMock

import numpy as np

from livecellx.core.single_cell import SingleCellStatic
from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset


class SingleCellStaticTestCase(unittest.TestCase):
    def setUp(self):
        self.single_cell = SingleCellStatic()
        self.single_cell.id = 1
        self.single_cell.timeframe = 0
        # self.single_cell.mask_dataset = MagicMock()
        self.mask = np.zeros([5, 5])
        self.mask_dataset = SingleImageDataset(img=self.mask)
        self.contour = np.array([[0, 2], [2, 2], [2, 0], [0, 0]])
        self.single_cell.img_dataset = SingleImageDataset(img=np.zeros([5, 5]))
        self.single_cell.mask_dataset = self.mask_dataset
        self.single_cell.contour = self.contour

    def test_get_mask_with_mask_dataset(self):
        mask = self.single_cell.get_mask()
        self.assertTrue((mask == self.mask).all())
        self.assertTrue(mask.shape == (5, 5))

    def test_get_mask_with_contour(self):
        self.single_cell.contour = self.contour
        _mask_dataset = self.single_cell.mask_dataset
        self.single_cell.mask_dataset = None
        mask = self.single_cell.get_mask()
        self.assertIsNotNone(mask)
        self.single_cell.mask_dataset = _mask_dataset

    def test_get_mask_with_no_mask_dataset_and_contour(self):
        self.single_cell.mask_dataset = None
        self.single_cell.contour = None
        with self.assertRaises(ValueError):
            self.single_cell.get_mask()


if __name__ == "__main__":
    unittest.main()
