from livecell_tracker.core.datasets import LiveCellImageDataset
from livecell_tracker.core import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection
from conftest import TestUtils
from pathlib import PurePosixPath


def test_posix_pathlib_convert_windows_path():
    p = PurePosixPath("c:\\windows")
    assert p.as_posix() == "c:/windows", "pathlib bug?..."


def test_posix_paths():
    imgs = LiveCellImageDataset(TestUtils.RAW_DATASET_PATH, ext="png")
    print("dataset path: %s" % (imgs.get_dataset_path()))
    print("dataset name: %s" % (imgs.get_dataset_name()))
    print("dataset img list: %s" % str(imgs.time2path))
    all_paths_cat = "**".join(imgs.time2path)
    assert all_paths_cat.find("\\") == -1


# TODO
def test_LiveCellImageDataset_json():
    pass


# TODO
def test_SingleCell_json():
    sc = SingleCellStatic()
    pass


# TODO
def test_SingleCellTrajectory_json():
    sc = SingleCellTrajectory()
    pass


# TODO
def test_SingleCellTrajectoryCollection_json():
    sc = SingleCellTrajectoryCollection()
    pass


if __name__ == "__main__":
    # test_posix_pathlib()
    test_posix_paths()
