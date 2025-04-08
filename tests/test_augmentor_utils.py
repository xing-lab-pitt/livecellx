import unittest
import numpy as np

from livecellx.preprocess.augmentor_utils import augment_images_by_augmentor


class TestAugmentorUtils(unittest.TestCase):
    """Test cases for augmentor utility functions in preprocess.augmentor_utils."""

    def setUp(self):
        """Set up test data."""
        # Create sample images and masks
        self.images = np.random.rand(5, 100, 100, 1) * 255
        self.masks = np.zeros((5, 100, 100, 1), dtype=np.uint8)
        for i in range(5):
            self.masks[i, 30:70, 30:70, 0] = 1

    @unittest.skip("Requires Augmentor package and is computationally expensive")
    def test_augment_images_by_augmentor(self):
        """Test augment_images_by_augmentor function."""
        # Set augmentation parameters
        crop_image_size = 64
        sampling_amount = 10

        try:
            # Augment images
            augmented_images, augmented_masks = augment_images_by_augmentor(
                self.images, self.masks, crop_image_size, sampling_amount
            )

            # Check output shapes
            self.assertEqual(len(augmented_images), sampling_amount)
            self.assertEqual(len(augmented_masks), sampling_amount)
            self.assertEqual(augmented_images.shape[1:3], (crop_image_size, crop_image_size))
            self.assertEqual(augmented_masks.shape[1:3], (crop_image_size, crop_image_size))

            # Check that masks contain only values 0 and 255 (binary masks might be scaled to 255)
            unique_values = np.unique(augmented_masks)
            self.assertTrue(len(unique_values) <= 2, f"Found more than 2 unique values in masks: {unique_values}")
            if len(unique_values) == 2:
                self.assertTrue(unique_values[0] == 0, f"First unique value is not 0: {unique_values[0]}")
                self.assertTrue(unique_values[1] > 0, f"Second unique value is not positive: {unique_values[1]}")

            # Check that images are properly augmented (different from original)
            self.assertFalse(np.array_equal(augmented_images[0], self.images[0][:crop_image_size, :crop_image_size]))
        except ImportError:
            # Skip test if Augmentor is not available
            self.skipTest("Augmentor package not available")


if __name__ == "__main__":
    unittest.main()
