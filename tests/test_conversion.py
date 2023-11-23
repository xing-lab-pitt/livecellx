import os
import shutil
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
from livecellx.preprocess.conversion import convert_livecell_dataset


class TestConvertLiveCellDataset(unittest.TestCase):
    def setUp(self):
        self.out_dir = Path("_unittest_tmp_conversion_output")
        os.makedirs(self.out_dir, exist_ok=True)

    def tearDown(self):
        if self.out_dir.is_dir():
            shutil.rmtree(self.out_dir)

    def test_convert_with_default_options(self):
        # Create a mock dataset
        class MockDataset:
            def __init__(self):
                self.times = [0, 1, 2]
                self.time2url = {0: "image_0.png", 1: "image_1.png", 2: "image_2.png"}
                self.ext = "png"

            def get_img_by_time(self, time):
                # Return a mock image
                return np.zeros((100, 100), dtype=np.uint8)

        # Set up the test parameters
        dataset = MockDataset()
        times = None
        filename_pattern = None
        keep_original_filename = True
        overwrite = False

        # Call the function under test
        convert_livecell_dataset(dataset, self.out_dir, times, filename_pattern, keep_original_filename, overwrite)

        # Check if the images are saved correctly
        expected_files = ["image_0.png", "image_1.png", "image_2.png"]
        for filename in expected_files:
            out_path = Path(self.out_dir) / Path(filename)
            self.assertTrue(out_path.exists())

    def test_convert_with_custom_options(self):
        # Create a mock dataset
        class MockDataset:
            def __init__(self):
                self.times = [0, 1, 2]
                self.time2url = {0: "image_0.png", 1: "image_1.png", 2: "image_2.png"}
                self.ext = "png"

            def get_img_by_time(self, time):
                # Return a mock image
                return np.zeros((100, 100), dtype=np.uint8)

        # Set up the test parameters
        dataset = MockDataset()
        times = [1, 2]
        filename_pattern = "img_{time}.png"
        keep_original_filename = False
        overwrite = True

        # Call the function under test
        convert_livecell_dataset(dataset, self.out_dir, times, filename_pattern, keep_original_filename, overwrite)

        # Check if the images are saved correctly
        expected_files = ["1.png", "2.png"]
        for filename in expected_files:
            out_path = Path(self.out_dir) / Path(filename)
            self.assertTrue(out_path.exists())

    def test_convert_with_existing_files(self):
        # Create a mock dataset
        class MockDataset:
            def __init__(self):
                self.times = [0, 1, 2]
                self.time2url = {0: "image_0.png", 1: "image_1.png", 2: "image_2.png"}
                self.ext = "png"

            def get_img_by_time(self, time):
                # Return a mock image
                return np.zeros((100, 100), dtype=np.uint8)

        # Set up the test parameters
        dataset = MockDataset()
        times = [0, 1]
        filename_pattern = None
        keep_original_filename = True
        overwrite = False

        # Create existing files
        existing_files = ["image_1.png", "image_0.png"]
        for filename in existing_files:
            out_path = Path(self.out_dir) / Path(filename)
            out_path.touch()

        # Call the function under test
        convert_livecell_dataset(dataset, self.out_dir, times, filename_pattern, keep_original_filename, overwrite)

        # Check if the existing files are skipped
        skipped_files = ["image_2.png"]
        for filename in skipped_files:
            out_path = Path(self.out_dir) / Path(filename)
            self.assertFalse(out_path.exists())

    def test_convert_with_overwrite_option(self):
        # Create a mock dataset
        class MockDataset:
            def __init__(self):
                self.times = [0, 1, 2]
                self.time2url = {0: "image_0.png", 1: "image_1.png", 2: "image_2.png"}
                self.ext = "png"

            def get_img_by_time(self, time):
                # Return a mock image
                return np.zeros((100, 100), dtype=np.uint8)

        # Set up the test parameters
        dataset = MockDataset()
        times = None
        filename_pattern = None
        keep_original_filename = True
        overwrite = True

        # Create existing files
        existing_files = ["0.png", "1.png"]
        for filename in existing_files:
            out_path = Path(self.out_dir) / Path(filename)
            out_path.touch()

        # Call the function under test
        convert_livecell_dataset(dataset, self.out_dir, times, filename_pattern, keep_original_filename, overwrite)

        # Check if the existing files are overwritten
        overwritten_files = ["0.png", "1.png"]
        for filename in overwritten_files:
            out_path = Path(self.out_dir) / Path(filename)
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
