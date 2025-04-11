import unittest
import numpy as np
import cv2 as cv
from livecellx.preprocess.utils import overlay_by_color, normalize_img_to_uint8


class TestOverlayByColor(unittest.TestCase):
    """Test cases for overlay_by_color function in preprocess.utils."""

    def setUp(self):
        """Set up test data."""
        # Create a sample image and mask
        self.img = np.random.rand(100, 100) * 255
        self.mask = np.zeros((100, 100), dtype=np.uint8)
        self.mask[30:70, 30:70] = 1

    def test_overlay_by_color(self):
        """Test overlay_by_color function with grayscale image."""
        # Create overlay with default parameters
        overlay_img = overlay_by_color(self.img, self.mask)

        # Check that output is a numpy array
        self.assertIsInstance(overlay_img, np.ndarray)

        # Check dimensions and channels (should be BGR)
        self.assertEqual(overlay_img.shape, (*self.img.shape, 3))

        # Check that output is uint8
        self.assertEqual(overlay_img.dtype, np.uint8)

        # Test with custom color and alpha
        color = (0, 0, 255)  # Red in BGR
        alpha = 0.7
        overlay_img_custom = overlay_by_color(self.img, self.mask, color=color, alpha=alpha)
        self.assertIsInstance(overlay_img_custom, np.ndarray)

        # Check that masked area has the specified color influence
        mask_region = overlay_img_custom[self.mask > 0]
        self.assertTrue(np.any(mask_region[:, 2] > mask_region[:, 0]))  # Red channel should be stronger

        # Test with custom normalization function
        def custom_normalize(img):
            # Simple min-max normalization
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                normalized = (img - img_min) / (img_max - img_min) * 255
            else:
                normalized = img - img_min
            return normalized.astype(np.uint8)

        overlay_img_custom_norm = overlay_by_color(self.img, self.mask, normalize_func=custom_normalize)
        self.assertIsInstance(overlay_img_custom_norm, np.ndarray)
        self.assertEqual(overlay_img_custom_norm.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
