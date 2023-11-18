import unittest
from livecellx.track.classify_utils import insert_time_segments


class TestInsertTimeSegments(unittest.TestCase):
    def test_empty_list(self):
        new_segment = (0, 1)
        disjoint_segments = []
        expected_output = [(0, 1)]
        self.assertEqual(insert_time_segments(new_segment, disjoint_segments), expected_output)

    def test_no_overlap(self):
        new_segment = (2, 3)
        disjoint_segments = [(0, 1), (4, 5)]
        expected_output = [(0, 1), (2, 3), (4, 5)]
        self.assertEqual(insert_time_segments(new_segment, disjoint_segments), expected_output)

    def test_overlap(self):
        new_segment = (2, 4)
        disjoint_segments = [(0, 1), (3, 5)]
        expected_output = [(0, 1), (2, 5)]
        self.assertEqual(insert_time_segments(new_segment, disjoint_segments), expected_output)

    def test_multiple_overlaps(self):
        new_segment = (2, 4)
        disjoint_segments = [(0, 1), (3, 5), (6, 7), (2, 8)]
        expected_output = [(0, 1), (2, 8)]
        self.assertEqual(insert_time_segments(new_segment, disjoint_segments), expected_output)

    def test_already_included(self):
        new_segment = (2, 4)
        disjoint_segments = [(0, 1), (2, 5)]
        expected_output = [(0, 1), (2, 5)]
        self.assertEqual(insert_time_segments(new_segment, disjoint_segments), expected_output)

    def test_already_included_multiple(self):
        new_segment = (2, 4)
        disjoint_segments = [(0, 1), (2, 5)]
        expected_output = [(0, 1), (2, 5)]
        self.assertEqual(insert_time_segments(new_segment, disjoint_segments), expected_output)

    def test_already_included_overlap(self):
        new_segment = (2, 4)
        disjoint_segments = [(0, 1), (2, 5), (4, 6)]
        expected_output = [(0, 1), (2, 6)]
        self.assertEqual(insert_time_segments(new_segment, disjoint_segments), expected_output)
