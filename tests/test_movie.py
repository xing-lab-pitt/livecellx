from pathlib import Path
import shutil
import unittest
from unittest.mock import Mock

import numpy as np
from livecellx.track.movie import generate_scs_movie
from livecellx.core.datasets import LiveCellImageDataset


class TestGenerateScsMovie(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path("./tmp_movie/")
        # Mock SingleCellStatic objects
        self.sc1 = Mock()
        self.sc2 = Mock()
        self.sc3 = Mock()
        self.sc4 = Mock()
        self.sc5 = Mock()

        # Mock get_sc_mask method to return specific masks
        self.sc1.get_sc_mask.return_value = np.array([[True, False], [False, True]])
        self.sc2.get_sc_mask.return_value = np.array([[True, True], [False, False]])
        self.sc3.get_sc_mask.return_value = np.array([[False, False], [True, True]])
        self.sc4.get_sc_mask.return_value = np.array([[True, True], [True, True]])
        self.sc5.get_sc_mask.return_value = np.array([[False, False], [False, False]])

    def test_generate_scs_movie(self):
        # Test case where there are SingleCellStatic objects and image dataset
        scs = [self.sc1, self.sc2, self.sc3, self.sc4, self.sc5]
        img_dataset = LiveCellImageDataset(
            time2url={
                0: "https://livecellimagestorage.blob.core.windows.net/images/2020-10-08/2020-10-08_14-00-00_000.png",
                1: "https://livecellimagestorage.blob.core.windows.net/images/2020-10-08/2020-10-08_14-00-00_001.png",
                2: "https://livecellimagestorage.blob.core.windows.net/images/2020-10-08/2020-10-08_14-00-00_002.png",
                3: "https://livecellimagestorage.blob.core.windows.net/images/2020-10-08/2020-10-08_14-00-00_003.png",
                4: "https://livecellimagestorage.blob.core.windows.net/images/2020-10-08/2020-10-08_14-00-00_004.png",
            },
            read_img_url_func=lambda x: np.zeros((2, 2)),
        )
        save_dir = self.tmp_dir / Path("test_generate_scs_movie/")
        fps = 3
        factor = 0.5
        video_only = False
        use_all_imgs = True

        # Call the function
        generate_scs_movie(scs, img_dataset, save_dir, fps, factor, video_only, use_all_imgs)

        # Add your assertions here to verify the expected behavior of the function
        # Check save_dir / imgs exists
        # Check save_dir / movie.mp4 exists

        self.assertTrue(Path(save_dir / "imgs").exists())
        self.assertTrue(Path(save_dir / "movie.mp4").exists())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)


if __name__ == "__main__":
    unittest.main()
