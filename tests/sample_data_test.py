import unittest
import os
import shutil
from pathlib import Path
from livecellx import sample_data
from livecellx.core.datasets import LiveCellImageDataset


class TestTutorialThreeImageSys(unittest.TestCase):
    def setUp(self):
        self.dic_dataset_path = Path("./datasets/test_data_STAV-A549/DIC_data")
        self.mask_dataset_path = Path("./datasets/test_data_STAV-A549/mask_data")

    def test_tutorial_three_image_sys(self):
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys(self.dic_dataset_path, self.mask_dataset_path)

        # Check that the .zip file was downloaded
        zip_file_path = Path(sample_data.DEFAULT_DATA_DIR) / "test_data_STAV-A549.zip"
        self.assertTrue(zip_file_path.is_file(), "Zip file was not downloaded")

        # Check that the datasets are instances of LiveCellImageDataset
        self.assertIsInstance(dic_dataset, LiveCellImageDataset)
        self.assertIsInstance(mask_dataset, LiveCellImageDataset)

        # Check that the datasets have the right length (3 images)
        self.assertEqual(len(dic_dataset), 3)
        self.assertEqual(len(mask_dataset), 3)

    def tearDown(self):
        # Clean up the downloaded zip file and the extracted datasets
        zip_file_path = Path(sample_data.DEFAULT_DATA_DIR) / "test_data_STAV-A549.zip"
        if zip_file_path.is_file():
            os.remove(zip_file_path)
        if self.dic_dataset_path.is_dir():
            shutil.rmtree(self.dic_dataset_path)
        if self.mask_dataset_path.is_dir():
            shutil.rmtree(self.mask_dataset_path)


if __name__ == "__main__":
    unittest.main()
