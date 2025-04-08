import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt

from livecellx.preprocess.correct_bg import (
    correct_background_median_gamma,
    correct_background_bisplrep,
    correct_background_polyfit,
)


class TestBackgroundCorrection(unittest.TestCase):
    """Test cases for background correction functions in preprocess.correct_bg."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample image with uneven background
        x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        background = 50 * (x + y)  # Gradient background
        signal = np.zeros((100, 100))
        signal[40:60, 40:60] = 100  # Signal in the center
        self.img = background + signal
    
    def test_correct_background_polyfit(self):
        """Test correct_background_polyfit function."""
        # Correct background
        corrected_img = correct_background_polyfit(self.img)
        
        # Check that output is a numpy array
        self.assertIsInstance(corrected_img, np.ndarray)
        
        # Check dimensions
        self.assertEqual(corrected_img.shape, self.img.shape)
        
        # Check that output is non-negative
        self.assertTrue(np.all(corrected_img >= 0))
        
        # Test with custom degree
        degree = 3
        corrected_img_custom = correct_background_polyfit(self.img, degree=degree)
        self.assertIsInstance(corrected_img_custom, np.ndarray)
        
        # Check that signal is preserved (center should be brighter than surroundings)
        center_intensity = np.mean(corrected_img[40:60, 40:60])
        surrounding_intensity = np.mean(corrected_img) - center_intensity
        self.assertGreater(center_intensity, surrounding_intensity)
    
    def test_correct_background_bisplrep(self):
        """Test correct_background_bisplrep function."""
        # Correct background
        corrected_img = correct_background_bisplrep(self.img)
        
        # Check that output is a numpy array
        self.assertIsInstance(corrected_img, np.ndarray)
        
        # Check dimensions
        self.assertEqual(corrected_img.shape, self.img.shape)
        
        # Check that output is non-negative
        self.assertTrue(np.all(corrected_img >= 0))
        
        # Test with custom parameters
        corrected_img_custom = correct_background_bisplrep(self.img, kx=4, ky=4, s=1e10, sample_step=2)
        self.assertIsInstance(corrected_img_custom, np.ndarray)
        
        # Check that signal is preserved (center should be brighter than surroundings)
        center_intensity = np.mean(corrected_img[40:60, 40:60])
        surrounding_intensity = np.mean(corrected_img) - center_intensity
        self.assertGreater(center_intensity, surrounding_intensity)
    
    def test_correct_background_median_gamma(self):
        """Test correct_background_median_gamma function."""
        try:
            # Correct background
            corrected_img = correct_background_median_gamma(self.img.astype(np.uint8))
            
            # Check that output is a numpy array
            self.assertIsInstance(corrected_img, np.ndarray)
            
            # Check dimensions
            self.assertEqual(corrected_img.shape, self.img.shape)
            
            # Test with custom disk size
            disk_size = 10
            corrected_img_custom = correct_background_median_gamma(self.img.astype(np.uint8), disk_size=disk_size)
            self.assertIsInstance(corrected_img_custom, np.ndarray)
            
            # Check that signal is preserved (center should be brighter than surroundings)
            center_intensity = np.mean(corrected_img[40:60, 40:60])
            surrounding_intensity = np.mean(corrected_img) - center_intensity
            self.assertGreater(center_intensity, surrounding_intensity)
        except ImportError:
            # Skip test if skimage is not available
            self.skipTest("skimage not available")


if __name__ == "__main__":
    unittest.main()
