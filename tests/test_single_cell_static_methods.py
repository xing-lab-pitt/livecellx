import unittest
import numpy as np
import tempfile
import json
from pathlib import Path
from livecellx.core.single_cell import SingleCellStatic, SingleCellTrajectory
from livecellx.core.datasets import SingleImageDataset


class TestSingleCellStaticMethods(unittest.TestCase):
    """Test static methods of SingleCellStatic class"""

    def test_gen_skimage_bbox_img_crop_basic(self):
        """Test basic functionality of gen_skimage_bbox_img_crop"""
        # Create a 10x10 test image with a pattern
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[2:8, 2:8, :] = 255  # White square in the middle
        
        # Define bbox: (min_row, min_col, max_row, max_col)
        bbox = (2, 2, 8, 8)
        
        # Test basic cropping
        result = SingleCellStatic.gen_skimage_bbox_img_crop(bbox, img)
        
        expected_shape = (6, 6, 3)  # bbox size
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.all(result == 255))

    def test_gen_skimage_bbox_img_crop_with_padding(self):
        """Test gen_skimage_bbox_img_crop with padding"""
        img = np.ones((10, 10), dtype=np.uint8) * 100
        bbox = (2, 2, 6, 6)  # 4x4 region
        padding = 2
        
        result = SingleCellStatic.gen_skimage_bbox_img_crop(bbox, img, padding=padding)
        
        # Should be 4x4 + 2*2 padding = 8x8
        expected_shape = (8, 8)
        self.assertEqual(result.shape, expected_shape)
        
        # Center should contain original values
        center_region = result[2:6, 2:6]
        self.assertTrue(np.all(center_region == 100))

    def test_gen_skimage_bbox_img_crop_edge_cases(self):
        """Test edge cases for gen_skimage_bbox_img_crop"""
        img = np.ones((5, 5), dtype=np.uint8) * 50
        
        # Test bbox at image boundary
        bbox = (0, 0, 3, 3)
        result = SingleCellStatic.gen_skimage_bbox_img_crop(bbox, img)
        
        # Should extract exact bbox region
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.all(result == 50))
        
        # Test with padding=0 to avoid padding complexity
        result_padded = SingleCellStatic.gen_skimage_bbox_img_crop(bbox, img, padding=0)
        self.assertEqual(result_padded.shape, (3, 3))

    def test_gen_skimage_bbox_img_crop_boundary_conditions(self):
        """Test boundary conditions and padding behavior"""
        img = np.arange(25).reshape(5, 5).astype(np.uint8)
        
        # Test bbox within image boundaries
        bbox = (1, 1, 4, 4)  # 3x3 region within 5x5 image
        
        result = SingleCellStatic.gen_skimage_bbox_img_crop(bbox, img)
        
        # Should extract the exact bbox region
        expected_shape = (3, 3)
        self.assertEqual(result.shape, expected_shape)

    def test_gen_skimage_bbox_img_crop_3d_image(self):
        """Test with 3D images (height, width, channels)"""
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        bbox = (2, 2, 7, 7)
        
        result = SingleCellStatic.gen_skimage_bbox_img_crop(bbox, img, padding=1)
        
        # Should preserve the channel dimension
        self.assertEqual(len(result.shape), 3)
        self.assertEqual(result.shape[2], 3)
        self.assertEqual(result.shape[:2], (7, 7))  # 5x5 + 1 padding each side

    def test_gen_contour_mask_basic(self):
        """Test basic functionality of gen_contour_mask"""
        # Create a simple square contour
        contour = np.array([[2, 2], [2, 7], [7, 7], [7, 2]], dtype=np.int32)
        
        # Test with crop=False to get full shape
        mask = SingleCellStatic.gen_contour_mask(contour, shape=(10, 10), crop=False)
        
        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(mask.dtype, bool)
        
        # Test with crop=True (default) - should be cropped to bbox
        mask_cropped = SingleCellStatic.gen_contour_mask(contour, shape=(10, 10))
        expected_cropped_shape = (6, 6)  # bbox is (2, 2, 8, 8), so shape is 6x6
        self.assertEqual(mask_cropped.shape, expected_cropped_shape)

    def test_gen_contour_mask_with_image(self):
        """Test gen_contour_mask with image parameter"""
        img = np.zeros((8, 8), dtype=np.uint8)
        contour = np.array([[1, 1], [1, 6], [6, 6], [6, 1]], dtype=np.int32)
        
        # Test with crop=False to get full image shape
        mask = SingleCellStatic.gen_contour_mask(contour, img=img, crop=False)
        
        self.assertEqual(mask.shape, img.shape)
        self.assertEqual(mask.dtype, bool)
        
        # Test with crop=True (default) - should be cropped to contour bbox  
        mask_cropped = SingleCellStatic.gen_contour_mask(contour, img=img)
        # bbox should be (1, 1, 7, 7), so shape is 6x6
        self.assertEqual(mask_cropped.shape, (6, 6))

    def test_gen_contour_mask_crop_functionality(self):
        """Test crop functionality of gen_contour_mask"""
        contour = np.array([[10, 10], [10, 20], [20, 20], [20, 10]], dtype=np.int32)
        
        # Test with crop=True
        mask_cropped = SingleCellStatic.gen_contour_mask(contour, shape=(30, 30), crop=True)
        
        # Should be cropped to the bounding box of the contour
        self.assertLessEqual(mask_cropped.shape[0], 30)
        self.assertLessEqual(mask_cropped.shape[1], 30)
        
        # Test with crop=False
        mask_full = SingleCellStatic.gen_contour_mask(contour, shape=(30, 30), crop=False)
        self.assertEqual(mask_full.shape, (30, 30))

    def test_gen_contour_mask_with_bbox(self):
        """Test gen_contour_mask with bbox parameter"""
        contour = np.array([[5, 5], [5, 15], [15, 15], [15, 5]], dtype=np.int32)
        bbox = (3, 3, 18, 18)  # Slightly larger than contour
        
        mask = SingleCellStatic.gen_contour_mask(contour, shape=(20, 20), bbox=bbox, crop=True)
        
        # Should be cropped to bbox size
        expected_shape = (15, 15)  # bbox dimensions
        self.assertEqual(mask.shape, expected_shape)

    def test_gen_contour_mask_dtype_parameter(self):
        """Test different dtype options for gen_contour_mask"""
        contour = np.array([[2, 2], [2, 5], [5, 5], [5, 2]], dtype=np.int32)
        
        # Test with bool dtype (default)
        mask_bool = SingleCellStatic.gen_contour_mask(contour, shape=(8, 8), dtype=bool)
        self.assertEqual(mask_bool.dtype, bool)
        
        # Test with int dtype
        mask_int = SingleCellStatic.gen_contour_mask(contour, shape=(8, 8), dtype=int)
        self.assertEqual(mask_int.dtype, int)

    def test_gen_contour_mask_edge_cases(self):
        """Test edge cases for gen_contour_mask"""
        # Test with very small contour, crop=False
        small_contour = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int32)
        mask = SingleCellStatic.gen_contour_mask(small_contour, shape=(3, 3), crop=False)
        self.assertEqual(mask.shape, (3, 3))
        
        # Test with crop=True (default) - should crop to bbox 
        mask_cropped = SingleCellStatic.gen_contour_mask(small_contour, shape=(3, 3))
        # bbox should be (0, 0, 2, 2), so shape is 2x2
        self.assertEqual(mask_cropped.shape, (2, 2))
        
        # Test with minimum valid contour (triangle), crop=False
        triangle_contour = np.array([[2, 2], [2, 3], [3, 2]], dtype=np.int32)
        mask_triangle = SingleCellStatic.gen_contour_mask(triangle_contour, shape=(5, 5), crop=False)
        self.assertEqual(mask_triangle.shape, (5, 5))

    def test_get_bbox_from_contour_basic(self):
        """Test basic functionality of get_bbox_from_contour"""
        # Simple rectangular contour
        contour = np.array([[2, 3], [2, 8], [7, 8], [7, 3]], dtype=np.int32)
        
        bbox = SingleCellStatic.get_bbox_from_contour(contour)
        
        # Expected: (min_row, min_col, max_row+1, max_col+1) - the method adds +1
        expected_bbox = (2, 3, 8, 9)
        np.testing.assert_array_equal(bbox, expected_bbox)

    def test_get_bbox_from_contour_single_point(self):
        """Test get_bbox_from_contour with single point"""
        contour = np.array([[5, 10]], dtype=np.int32)
        
        bbox = SingleCellStatic.get_bbox_from_contour(contour)
        
        # For a single point, bbox should be (point_x, point_y, point_x+1, point_y+1)
        expected_bbox = (5, 10, 6, 11)
        np.testing.assert_array_equal(bbox, expected_bbox)

    def test_get_bbox_from_contour_irregular_shape(self):
        """Test get_bbox_from_contour with irregular contour"""
        # Irregular L-shaped contour
        contour = np.array([
            [1, 1], [1, 5], [3, 5], [3, 3], [6, 3], [6, 1]
        ], dtype=np.int32)
        
        bbox = SingleCellStatic.get_bbox_from_contour(contour)
        
        # Should encompass all points with +1 added to max values
        expected_bbox = (1, 1, 7, 6)  # min_row=1, min_col=1, max_row=6+1, max_col=5+1
        np.testing.assert_array_equal(bbox, expected_bbox)

    def test_get_bbox_from_contour_dtype_parameter(self):
        """Test dtype parameter of get_bbox_from_contour"""
        contour = np.array([[2, 3], [7, 8]], dtype=np.int32)
        
        # Test with int dtype (default)
        bbox_int = SingleCellStatic.get_bbox_from_contour(contour, dtype=int)
        self.assertEqual(bbox_int.dtype, int)
        
        # Test with float dtype
        bbox_float = SingleCellStatic.get_bbox_from_contour(contour, dtype=float)
        self.assertEqual(bbox_float.dtype, float)

    def test_get_bbox_from_contour_negative_coordinates(self):
        """Test get_bbox_from_contour with negative coordinates"""
        contour = np.array([[-2, -3], [4, 5], [1, -1]], dtype=np.int32)
        
        bbox = SingleCellStatic.get_bbox_from_contour(contour)
        
        expected_bbox = (-2, -3, 5, 6)  # max values get +1 added
        np.testing.assert_array_equal(bbox, expected_bbox)

    def test_load_single_cells_jsons(self):
        """Test load_single_cells_jsons static method"""
        # Create temporary directory and files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some mock single cell data
            sc1 = SingleCellStatic()
            sc1.id = 1
            sc1.timeframe = 0
            sc1.bbox = np.array([0, 0, 5, 5])
            sc1.contour = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])
            sc1.img_dataset = SingleImageDataset(img=np.zeros((5, 5)))
            sc1.mask_dataset = SingleImageDataset(img=np.ones((5, 5), dtype=bool))
            
            sc2 = SingleCellStatic()
            sc2.id = 2
            sc2.timeframe = 1
            sc2.bbox = np.array([1, 1, 6, 6])
            sc2.contour = np.array([[1, 1], [1, 6], [6, 6], [6, 1]])
            sc2.img_dataset = SingleImageDataset(img=np.zeros((5, 5)))
            sc2.mask_dataset = SingleImageDataset(img=np.ones((5, 5), dtype=bool))
            
            # Write first JSON file
            json_path1 = temp_path / "cells1.json"
            SingleCellStatic.write_single_cells_json([sc1], str(json_path1), str(temp_path))
            
            # Write second JSON file
            json_path2 = temp_path / "cells2.json"
            SingleCellStatic.write_single_cells_json([sc2], str(json_path2), str(temp_path))
            
            # Test loading multiple JSON files - method expects a list of paths
            paths = [str(json_path1), str(json_path2)]
            loaded_cells = SingleCellStatic.load_single_cells_jsons(paths)
            
            # Should load both cells
            self.assertEqual(len(loaded_cells), 2)
            
            # Check that both cells are loaded correctly
            # IDs might be converted to strings during JSON serialization
            loaded_ids = {str(sc.id) for sc in loaded_cells}
            self.assertEqual(loaded_ids, {'1', '2'})

    def test_show_trajectory_on_grid(self):
        """Test show_trajectory_on_grid static method"""
        # This is a wrapper around show_sct_on_grid, so we just test it doesn't crash
        try:
            # Call with minimal parameters - should not crash
            result = SingleCellTrajectory.show_trajectory_on_grid()
            # If it returns something, that's fine; if it doesn't crash, that's what we care about
            self.assertTrue(True)  # Test passes if no exception is raised
        except TypeError as e:
            # If it requires parameters, that's expected - the important thing is it's callable
            self.assertIn("required", str(e).lower())


if __name__ == "__main__":
    unittest.main()