from PIL import Image, ImageSequence
import glob
import numpy as np


def save_png(img: np.array, path: str, mode="L"):
    """save image to png file

    Parameters
    ----------
    path : str
        path to save the image
    img : np.array
        image to save
    """
    img = img.astype(np.uint8)
    img = Image.fromarray(img, mode=mode)
    img.save(path)


def save_tiff(path: str, img: np.array):
    """save image to tiff file

    Parameters
    ----------
    path : str
        path to save the image
    img : np.array
        image to save
    """
    img = img.astype(np.uint8)
    img = Image.fromarray(img, mode="L")
    img.save(path)
