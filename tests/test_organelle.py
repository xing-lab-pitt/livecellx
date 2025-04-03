import unittest
import numpy as np
from pathlib import Path
from livecellx import sample_data
from livecellx.core.single_cell import Organelle, SingleCellStatic, SingleUnit
from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset
from tests.test_utils import TestHelper


class OrganelleTestCase(TestHelper):
    def setUp(self):
        # Create a basic image and mask dataset for testing
        self.mask = np.zeros((5, 5))
        self.img = np.zeros((5, 5))
        self.mask_dataset = SingleImageDataset(img=self.mask)
        self.img_dataset = SingleImageDataset(img=self.img)
        self.contour = np.array([[0, 2], [2, 2], [2, 0], [0, 0]])

        # Create an organelle instance
        self.organelle = Organelle(
            organelle_type="nucleus",
            parent_cell_id="cell123",
            timeframe=0,
            contour=self.contour,
            img_dataset=self.img_dataset,
            mask_dataset=self.mask_dataset,
        )
        self.organelle.id = "org123"

    def test_init(self):
        """Test the initialization of an Organelle instance"""
        self.assertEqual(self.organelle.organelle_type, "nucleus")
        self.assertEqual(self.organelle.parent_cell_id, "cell123")
        self.assertEqual(self.organelle.timeframe, 0)
        np.testing.assert_array_equal(self.organelle.contour, self.contour)
        self.assertEqual(self.organelle.img_dataset, self.img_dataset)
        self.assertEqual(self.organelle.mask_dataset, self.mask_dataset)
        self.assertEqual(self.organelle.id, "org123")

    def test_repr(self):
        """Test the string representation of an Organelle instance"""
        expected_repr = "Suborganelle(type=nucleus, id=org123, timeframe=0)"
        self.assertEqual(repr(self.organelle), expected_repr)

    def test_to_json_dict(self):
        """Test converting an Organelle instance to a JSON dictionary"""
        json_dict = self.organelle.to_json_dict()

        # Check organelle-specific fields
        self.assertEqual(json_dict["organelle_type"], "nucleus")
        self.assertEqual(json_dict["parent_cell_id"], "cell123")

        # Check inherited fields
        self.assertEqual(json_dict["timeframe"], 0)
        self.assertEqual(json_dict["id"], "org123")
        self.assertTrue(isinstance(json_dict["contour"], list))
        np.testing.assert_array_equal(np.array(json_dict["contour"]), self.contour)

    def test_load_from_json_dict(self):
        """Test loading an Organelle instance from a JSON dictionary"""
        # Create a JSON dictionary
        json_dict = self.organelle.to_json_dict()

        # Create a new organelle and load from the JSON dictionary
        new_organelle = Organelle(organelle_type="unknown")
        new_organelle.load_from_json_dict(json_dict, self.img_dataset, self.mask_dataset)

        # Check that the fields were loaded correctly
        self.assertEqual(new_organelle.organelle_type, "nucleus")
        self.assertEqual(new_organelle.parent_cell_id, "cell123")
        self.assertEqual(new_organelle.timeframe, 0)
        np.testing.assert_array_equal(new_organelle.contour, self.contour)
        self.assertEqual(new_organelle.id, "org123")

    def test_inheritance(self):
        """Test that Organelle inherits from SingleUnit"""
        self.assertIsInstance(self.organelle, SingleUnit)

        # Test that organelle can use SingleUnit methods
        mask = self.organelle.get_mask()
        self.assertTrue((mask == self.mask).all())


if __name__ == "__main__":
    unittest.main()
