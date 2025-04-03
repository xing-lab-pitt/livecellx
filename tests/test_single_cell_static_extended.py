import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from livecellx import sample_data
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.core.single_cell import SingleCellStatic
from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset
from tests.test_utils import TestHelper


class SingleCellStaticExtendedTestCase(TestHelper):
    @classmethod
    def setUpClass(cls):
        # Load sample data
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        cls.single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)
        cls.img_dataset = dic_dataset
        cls.mask_dataset = mask_dataset

    def setUp(self):
        # Use the first single cell for testing
        self.single_cell = self.single_cells[0]

    def test_copy(self):
        """Test copying a SingleCellStatic instance"""
        # Copy the single cell
        copied_cell = self.single_cell.copy()

        # Check that the copy has the same fields
        self.assertEqual(copied_cell.timeframe, self.single_cell.timeframe)
        np.testing.assert_array_equal(copied_cell.bbox, self.single_cell.bbox)
        np.testing.assert_array_equal(copied_cell.contour, self.single_cell.contour)
        self.assertEqual(copied_cell.feature_dict, self.single_cell.feature_dict)
        self.assertEqual(copied_cell.meta, self.single_cell.meta)
        self.assertEqual(copied_cell.id, self.single_cell.id)

        # Check that the copy is a different object
        self.assertIsNot(copied_cell, self.single_cell)

        # Modify the copy and check that the original is unchanged
        copied_cell.timeframe = 999
        self.assertNotEqual(copied_cell.timeframe, self.single_cell.timeframe)

    def test_compute_regionprops(self):
        """Test computing region properties for a single cell"""
        # Compute region properties
        props = self.single_cell.compute_regionprops()

        # Check that the properties are not None
        self.assertIsNotNone(props)

        # Check that the properties have the expected attributes
        self.assertTrue(hasattr(props, "area"))
        self.assertTrue(hasattr(props, "centroid"))
        self.assertTrue(hasattr(props, "bbox"))

        # Test with crop=False
        props_no_crop = self.single_cell.compute_regionprops(crop=False)

        # Check that the properties are not None
        self.assertIsNotNone(props_no_crop)

        # Test with ignore_errors=True
        props_ignore_errors = self.single_cell.compute_regionprops(ignore_errors=True)

        # Check that the properties are not None
        self.assertIsNotNone(props_ignore_errors)

    def test_get_napari_shapes(self):
        """Test getting Napari shapes for a single cell"""
        # Get the shape for bounding box
        bbox_shape = self.single_cell.get_napari_shape_bbox_vec()

        # Check that the shape is not None
        self.assertIsNotNone(bbox_shape)

        # Get the shape for contour
        contour_shape = self.single_cell.get_napari_shape_contour_vec()

        # Check that the shape is not None
        self.assertIsNotNone(contour_shape)

    def test_show_methods(self):
        """Test the show methods of SingleCellStatic"""
        # Test show method
        fig, ax = plt.subplots()
        self.single_cell.show(ax=ax)
        plt.close(fig)

        # Test show_mask method
        fig, ax = plt.subplots()
        self.single_cell.show_mask(ax=ax)
        plt.close(fig)

        # Test show_contour_img method
        fig, ax = plt.subplots()
        self.single_cell.show_contour_img(ax=ax)
        plt.close(fig)

        # Test show_contour_mask method
        fig, ax = plt.subplots()
        self.single_cell.show_contour_mask(ax=ax)
        plt.close(fig)

        # Test show_panel method
        axes = self.single_cell.show_panel()
        plt.close(plt.gcf())

        # Check that the axes are not None
        self.assertIsNotNone(axes)
        self.assertEqual(len(axes), 6)


if __name__ == "__main__":
    unittest.main()
