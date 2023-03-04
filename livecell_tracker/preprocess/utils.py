import glob
import os
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cellpose import models
from cellpose.io import imread
from PIL import Image, ImageSequence, ImageEnhance
from tqdm import tqdm
import cv2 as cv


def normalize_img_to_uint8(img: np.array, dtype=np.uint8) -> np.array:
    """calculate z score of img and normalize to range [0, 255]

    Parameters
    ----------
    img : np.array
        _description_

    Returns
    -------
    _type_
        _description_
    """
    img = (img - np.mean(img.flatten())) / np.std(img.flatten())
    img = img + abs(np.min(img.flatten()))
    img = img / np.max(img) * 255
    return img.astype(dtype)


def livetracker_standard_normalize(img):
    img = normalize_img_to_uint8(img)
    return img


def overlay(image, mask, mask_channel_rgb_val=100, img_channel_rgb_val_factor=1):
    mask = mask.astype(np.uint8)
    mask[mask > 0] = mask_channel_rgb_val
    image = normalize_img_to_uint8(image).astype(np.uint8)
    image = image * img_channel_rgb_val_factor
    res = np.zeros(list(mask.shape) + [3])
    res[:, :, 2] = image
    res[:, :, 1] = mask
    res = Image.fromarray(res.astype(np.uint8))
    return res


# TODO: add tests and note if scale * raw_image exceeds type boundaries such as 255
def reserve_img_by_pixel_percentile(raw_img: np.array, percentile: float, target_val: float = None, scale: float = 1):
    """
    Parameters
    ----------
    raw_img : np.array
        _description_
    percentile : float
        _description_
    target_val : float, optional
        _description_, by default None
    scale : float, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    flattened_img = raw_img.copy().flatten()
    is_above_threshold = flattened_img > np.percentile(flattened_img, percentile)
    if target_val is not None:
        flattened_img[is_above_threshold] = target_val
    elif scale is not None:
        flattened_img[is_above_threshold] *= scale
    else:
        raise ValueError("Must specify either target_val or scale")
    flattened_img[np.logical_not(is_above_threshold)] = 0
    return flattened_img.reshape(raw_img.shape)


def _enhance_contrast(im: Image, factor=5):
    enhancer = ImageEnhance.Contrast(im)
    im_output = enhancer.enhance(factor)
    return im_output


def enhance_contrast(img: np.array, factor=5):
    im = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(im)
    im_output = enhancer.enhance(factor)
    return np.array(im_output)


def dilate_or_erode_mask(cropped_mask: np.array, scale_factor):
    """
    # TODO ensure reproducibility with different values of padding
    cv's erode and dilation definition:  https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    scale factor < 0: erode
    scale factor = 0: do nothing but copy and return
    scale factor > 0: dilate
    """
    mask_area = np.sum(cropped_mask)
    mask_radius = np.sqrt(mask_area / np.pi)
    len_kernel = int(np.ceil(2 * mask_radius * np.abs(scale_factor) + 1))

    kernel = np.ones(shape=(len_kernel, len_kernel))
    if scale_factor < 0:
        s_cropped_mask = cv.erode(cropped_mask, kernel=kernel)
    elif scale_factor == 0:
        s_cropped_mask = cropped_mask.copy()
    else:  # quality_model_type_param > 0
        s_cropped_mask = cv.dilate(cropped_mask, kernel=kernel)
    return s_cropped_mask
