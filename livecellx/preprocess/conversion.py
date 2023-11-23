from pathlib import Path
from PIL import Image

from livecellx.core.datasets import LiveCellImageDataset
from livecellx.livecell_logger import main_info
from livecellx.preprocess.utils import normalize_img_to_uint8


def convert_livecell_dataset(
    dataset: LiveCellImageDataset,
    out_dir,
    times=None,
    filename_pattern=None,
    keep_original_filename=True,
    overwrite=False,
):
    """Save images to a directory"""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        main_info("Creating output directory %s" % out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    if times is None:
        times = dataset.times
    for time in times:
        img = dataset.get_img_by_time(time)
        img = normalize_img_to_uint8(img)
        # keep the original file names
        if filename_pattern is None and keep_original_filename:
            filename = Path(dataset.time2url[time]).name
        else:
            filename = Path("%d.%s" % (time, dataset.ext))
        out_path = Path(out_dir) / Path(filename)

        if out_path.exists() and not overwrite:
            main_info("Skip %s, already exists" % out_path)
            continue

        Image.fromarray(img).save(out_path)
