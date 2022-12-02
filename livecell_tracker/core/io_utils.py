from PIL import Image, ImageSequence
import glob
import numpy as np


def save_png(path: str, img: np.array, mode="L"):
    """save image to png file

    Parameters
    ----------
    path : str
        path to save the image
    img : np.array
        image to save
    """
    save_general(path=path, img=img, mode=mode)


def save_tiff(img: np.array, path: str, mode="L"):
    save_general(path=path, img=img, mode=mode)


def save_general(img: np.array, path: str, mode="L"):
    """save image to tiff file

    Parameters
    ----------
    path : str
        path to save the image
    img : np.array
        image to save
    """
    # TODO: discuss whether it is caller's responsibility to make img's mode correct
    if mode == "L" and np.unique(img).shape[0] >= 256:
        mode = "I"
        print(
            "Warning: saving image with more than 256 unique values as 8-bit, use I mode to save as 32-bit signed integer"
        )
    elif mode == "L" and (img < 0).any():
        mode = "I"
        print(
            "Warning: saving image with negative values as unsigned 8-bit, will use I mode to save as 32-bit signed integer instead"
        )
    elif mode == "L":
        img = img.astype(np.uint8)

    if mode == "I":
        img = Image.fromarray(img.astype(np.int32), mode=mode)
    else:
        img = Image.fromarray(img, mode=mode)
    img.save(path)
