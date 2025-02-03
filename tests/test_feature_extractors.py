import unittest
from unittest.mock import MagicMock, patch
from livecellx import sample_data
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.preprocess.utils import normalize_img_to_uint8
from livecellx.trajectory.feature_extractors import compute_skimage_regionprops, parallelize_compute_features
from livecellx.core.single_cell import SingleCellStatic


class TestParallelizeComputeFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        cls.cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)

    def setUp(self):
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        self.cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)

    def test_parallelize_compute_features_chunk(self):
        parallelize_compute_features(
            self.cells,
            compute_skimage_regionprops,
            params={
                "feature_key": "skimage",
                "preprocess_img_func": normalize_img_to_uint8,
                "sc_level_normalize": True,
                # "padding": PADDING,
                # "use_intensity": USE_INTENSITY,
                # "props": skimage_morph_properties,
                # "include_background": INCLUDE_BG,
            },
            cores=32,
        )
        for sc in self.cells:
            self.assertTrue(sc.feature_dict.get("skimage") is not None)

    def test_parallelize_compute_features_chunk_with_cores(self):
        parallelize_compute_features(
            self.cells,
            compute_skimage_regionprops,
            params={
                "feature_key": "skimage",
                "preprocess_img_func": normalize_img_to_uint8,
                "sc_level_normalize": True,
                # "padding": PADDING,
                # "use_intensity": USE_INTENSITY,
                # "props": skimage_morph_properties,
                # "include_background": INCLUDE_BG,
            },
            cores=1,
        )
        for sc in self.cells:
            self.assertTrue(sc.feature_dict.get("skimage") is not None)

    def test_parallelize_compute_features(self):
        parallelize_compute_features(
            self.cells,
            compute_skimage_regionprops,
            params={
                "feature_key": "skimage",
                "preprocess_img_func": normalize_img_to_uint8,
                "sc_level_normalize": True,
                # "padding": PADDING,
                # "use_intensity": USE_INTENSITY,
                # "props": skimage_morph_properties,
                # "include_background": INCLUDE_BG,
            },
            cores=32,
        )
        for sc in self.cells:
            self.assertTrue(sc.feature_dict.get("skimage") is not None)


if __name__ == "__main__":
    unittest.main()
