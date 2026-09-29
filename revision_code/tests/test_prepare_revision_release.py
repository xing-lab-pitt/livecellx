"""Small synthetic release checks; no research data or models are loaded."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "prepare_revision_release", Path(__file__).resolve().parents[1] / "prepare_revision_release.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous_root = release.ROOT
        release.ROOT = self.root
        self.out = self.root / "bundle"
        self.out.mkdir()
        self.builder = release.Release(self.out)

    def tearDown(self):
        release.ROOT = self.previous_root
        self.temp.cleanup()

    def test_recursive_paths_preserve_annotation_and_source(self):
        image = self.root / "image.tif"
        image.write_bytes(b"synthetic-image")
        ds = self.root / "dataset.json"
        ds.write_text(json.dumps({"data_dir_path": str(self.root), "time2url": {"0": str(image)}}))
        trajectory = self.root / "trajectory.json"
        content = {"meta": {"img_dataset_json_path": str(ds)}, "timeframe": 0,
                   "track_id": 168, "contour": [[1, 2], [3, 4], [5, 6]],
                   "feature_dict": {"area": 12.0}, "daughter_trajectory_ids": [169, 170]}
        trajectory.write_text(json.dumps(content))
        original = trajectory.read_bytes()
        self.builder.add(trajectory, "data", "test_collection", scan_json=True)
        self.builder.scan()
        normalized = json.loads(self.builder.files["trajectory.json"]["export"].read_text())
        content["meta"]["img_dataset_json_path"] = "dataset.json"
        self.assertEqual(normalized, content)
        self.assertEqual(trajectory.read_bytes(), original)
        self.assertEqual(set(self.builder.files), {"trajectory.json", "dataset.json", "image.tif"})
        self.builder.archive()
        release.verify(self.out)
        immutable = {p.name: release.sha256(p) for p in self.out.glob('*.tar.gz') if p.name != 'revision_code.tar.gz'}
        sources = self.root / 'revision_code'
        sources.mkdir()
        (sources / 'GITHUB_UPLOAD_FILES.txt').write_text('revision_code/example.py\n')
        (sources / 'example.py').write_text('value = 1\n')
        for name in ('pyproject.toml', 'requirements.txt', 'readme.md', 'LICENSE'):
            (self.root / name).write_text('synthetic metadata\n')
        with patch.object(release.subprocess, 'check_output', return_value=b''):
            release.refresh_code(self.out)
        for name, digest in immutable.items():
            self.assertEqual(release.sha256(self.out / name), digest)

        with (self.out / "revision_data.tar.gz").open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            release.verify(self.out)

    def test_missing_dependency_stops_build(self):
        trajectory = self.root / "trajectory.json"
        trajectory.write_text(json.dumps({"img_dataset_json_path": "missing.json"}))
        self.builder.add(trajectory, "data", "test_collection", scan_json=True)
        with self.assertRaises(FileNotFoundError):
            self.builder.scan()

    def test_outside_repository_is_not_silently_published(self):
        with self.assertRaises(ValueError):
            self.builder.relative(self.root.parent / "not_this_repository")


if __name__ == "__main__":
    unittest.main()
