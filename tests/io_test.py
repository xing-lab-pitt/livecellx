from livecellx.core.datasets import LiveCellImageDataset
from livecellx.core import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection
from .conftest import TestUtils
from pathlib import PurePosixPath, PureWindowsPath


def test_posix_pathlib_convert_windows_path():
    sample_path = PureWindowsPath("c:\\windows")
    assert sample_path.as_posix() == "c:/windows", (
        "pathlib bug? correct: c:/windows, actual: %s" % sample_path.as_posix()
    )


def test_posix_paths():
    imgs = LiveCellImageDataset(TestUtils.RAW_DATASET_PATH, ext="png")
    print(">>> debug, imgs:", imgs.get_dataset_name(), imgs.get_dataset_path())
    print("dataset path: %s" % (imgs.get_dataset_path()))
    print("dataset name: %s" % (imgs.get_dataset_name()))
    print("dataset img list: %s" % str(imgs.time2url))
    all_paths_cat = "**".join(imgs.time2url.values())
    assert all_paths_cat.find("\\") == -1


# TODO
def test_LiveCellImageDataset_json():
    pass


# TODO
def test_SingleCell_json():
    sc = SingleCellStatic(contour=[(1, 2), (3, 4)])
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
