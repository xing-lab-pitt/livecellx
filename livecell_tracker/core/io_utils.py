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
    save_general(path=path, img=img, mode=mode)


def save_tiff(path: str, img: np.array, mode="L"):
    save_general(path=path, img=img, mode=mode)


def save_general(path: str, img: np.array, mode="L"):
    """save image to tiff file

    Parameters
    ----------
    path : str
        path to save the image
    img : np.array
        image to save
    """
    if np.unique(img).size >= 256:
        mode = "I"
        print(
            "Warning: saving image with more than 256 unique values as 8-bit, use I mode to save as 32-bit signed integer"
        )
    else:
        img = img.astype(np.uint8)
    img = Image.fromarray(img, mode=mode)
    img.save(path)
