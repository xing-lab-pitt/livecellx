from pathlib import Path
from livecell_tracker.core.datasets import LiveCellImageDataset, SingleImageDataset


def tutorial_three_image(
    dic_dataset_path=Path("./datasets/test_data_STAV-A549/DIC_data"),
    mask_dataset_path=Path("./datasets/test_data_STAV-A549/mask_data"),
):
    mask_dataset = LiveCellImageDataset(mask_dataset_path, ext="png")
    dic_dataset = LiveCellImageDataset(dic_dataset_path, ext="tif")
    return dic_dataset, mask_dataset
