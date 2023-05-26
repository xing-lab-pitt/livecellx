from pathlib import Path
import os
import ntpath
import urllib
from urllib.request import urlretrieve
from livecell_tracker.core.datasets import LiveCellImageDataset, SingleImageDataset
from livecell_tracker.livecell_logger import LoggerManager, main_info
import zipfile

DEFAULT_DATA_DIR = Path("./datasets")


def extract_zip_data(filepath, dest):
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        main_info("Extracting data to " + str(dest))
        zip_ref.extractall(dest)


# TODO: use AWS S3 bucket to store the data instead
def tutorial_three_image_sys(
    dic_dataset_path=Path("./datasets/test_data_STAV-A549/DIC_data"),
    mask_dataset_path=Path("./datasets/test_data_STAV-A549/mask_data"),
    url="https://www.dropbox.com/s/p7gjpvgs0qop1ko/test_data_STAV-A549_v0.zip?dl=1",
    dir=DEFAULT_DATA_DIR,
):
    if dir is None:
        dir = DEFAULT_DATA_DIR
    zip_filepath = download_data(url, filename="test_data_STAV-A549.zip", dir=dir)
    extract_zip_data(filepath=zip_filepath, dest=dir)
    mask_dataset = LiveCellImageDataset(mask_dataset_path, ext="png")
    dic_dataset = LiveCellImageDataset(dic_dataset_path, ext="tif")
    return dic_dataset, mask_dataset


def download_data(url, filename=None, dir=DEFAULT_DATA_DIR):
    filepath = ntpath.basename(url) if filename is None else filename
    filepath = os.path.join(dir, filepath)
    main_info("Downloading data to " + filepath)

    if not os.path.exists(filepath):
        if not os.path.exists(dir):
            os.mkdir(dir)

        # download the data
        opener = urllib.request.build_opener()
        opener.addheaders = [
            (
                "User-agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
            )
        ]
        urllib.request.install_opener(opener)
        urlretrieve(url, filepath, reporthook=LoggerManager.get_main_logger().request_report_hook)
    else:
        main_info("Data already exists at " + filepath)

    return filepath
