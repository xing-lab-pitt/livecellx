import glob
import os
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageSequence, ImageEnhance
from scipy import ndimage
from tqdm import tqdm
import cv2 as cv
from livecellx.core.io_utils import save_png
from livecellx.preprocess.correct_bg import correct_background_bisplrep, correct_background_polyfit


def normalize_features_zscore(features: np.ndarray) -> np.ndarray:
    """normalize features to z score

    Parameters
    ----------
    features : np.ndarray
        _description_

    Returns
    -------
    _type_
        _description_
    """
    features = features - np.mean(features, axis=0)
    std = np.std(features, axis=0)
    if std != 0:
        features = features / std
    return features


# TODO: add tests
def normalize_img_by_bitdepth(img: np.ndarray, bit_depth: int = 8, mean=None) -> np.ndarray:
    """
    Normalize an image by its bit depth.
    Parameters:
    img (np.ndarray): Input image array.
    bit_depth (int, optional): Bit depth to normalize the image to. Must be one of 8, 16, or 32. Default is 8.
    mean (int, optional): Desired mean value for the normalized image. If None, defaults to half of the maximum value for the given bit depth.
    Returns:
    np.ndarray: Normalized image array.
    Raises:
    ValueError: If bit_depth is not one of 8, 16, or 32.
    """
    if bit_depth not in [8, 16, 32]:
        raise ValueError(f"bit_depth must be 8, 16, or 32, not {bit_depth}")

    img = img.astype(np.float32)  # Ensure floating point for normalization

    std = np.std(img.flatten())
    if std != 0:
        img = (img - np.mean(img.flatten())) / std
    else:
        img = img - np.mean(img.flatten())

    img = img + abs(np.min(img.flatten()))  # Shift to non-negative values

    if np.max(img.flatten()) != 0:
        img = img / np.max(img.flatten())  # Normalize to [0, 1]

    # Scale according to bit depth and mean to be half of the max val if not specified
    dtype = None
    dtype_max = None
    if bit_depth == 8:
        dtype_max = 255
    elif bit_depth == 16:
        dtype_max = 65535
    elif bit_depth == 32:
        dtype_max = np.iinfo(np.uint32).max
    else:
        raise ValueError(f"bit_depth must be 8, 16, or 32, not {bit_depth}")

    if bit_depth == 8:
        dtype = np.uint8
        img = img * 255
        mean = mean if mean is not None else 127
    elif bit_depth == 16:
        dtype = np.uint16
        img = img * 65535
        mean = mean if mean is not None else 32767
    elif bit_depth == 32:
        dtype = np.uint32
        img = img * np.iinfo(np.uint32).max  # Max uint32 value
        mean = mean if mean is not None else np.iinfo(np.uint32).max // 2
    else:
        raise ValueError(f"bit_depth must be 8, 16, or 32, not {bit_depth}")

    # Scale to mean
    mean = int(mean)

    img = img - np.mean(img.flatten()) + mean
    img = np.clip(img, 0, dtype_max)  # Ensure values are within [0, 255]
    img = img.astype(dtype)
    return img


def normalize_img_to_uint8(img: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """calculate z score of img and normalize to range [0, 255]

    Parameters
    ----------
    img : np.ndarray
        _description_

    Returns
    -------
    _type_
        _description_
    """
    std = np.std(img.flatten())
    if std != 0:
        img = (img - np.mean(img.flatten())) / std
    else:
        img = img - np.mean(img.flatten())
    img = img + abs(np.min(img.flatten()))
    if np.max(img) != 0:
        img = img / np.max(img) * 255
    return img.astype(dtype)


def standard_preprocess(img, bg_correct_func=correct_background_polyfit):
    if bg_correct_func is not None:
        img = bg_correct_func(img)
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


def overlay_by_color(image, mask, color=(100, 0, 0), alpha=0.5, normalize_func=None):
    """
    Overlay a color mask onto a grayscale or RGB image.

    Args:
        image (numpy.ndarray): Input image. If grayscale, it will be converted to BGR.
        mask (numpy.ndarray): Mask image, usually binary. Non-zero values indicate areas to overlay.
        color (tuple): Color to overlay in BGR format (default is (100, 0, 0), which is dark red).
        alpha (float): Opacity of the overlay (default is 0.5).
        normalize_func (callable, optional): Function to normalize the image. If None, normalize_img_to_uint8 is used.

    Returns:
        numpy.ndarray: Resulting image with the overlay in BGR format.
    """
    # Apply normalization if needed
    if normalize_func is None:
        normalize_func = normalize_img_to_uint8

    image = normalize_func(image)

    # Convert the grayscale image to BGR for overlaying
    if len(image.shape) == 2:
        image_bgr = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()

    # Ensure mask is binary and in uint8 format
    mask = mask.astype(np.uint8)
    if mask.max() > 1:
        # Normalize mask if it's not binary (0 and 1)
        _, mask = cv.threshold(mask, 0, 1, cv.THRESH_BINARY)

    # Create a color overlay image
    overlay = np.zeros_like(image_bgr, dtype=np.uint8)

    # Apply color to the mask regions in the overlay
    for c in range(3):  # BGR channels
        overlay[:, :, c] = mask * color[c]

    # Blend the overlay onto the image
    result = cv.addWeighted(image_bgr, 1.0, overlay, alpha, 0.0)

    # Ensure the result is in the valid range and correct type
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


# TODO: add tests and note if scale * raw_image exceeds type boundaries such as 255
def reserve_img_by_pixel_percentile(raw_img: np.ndarray, percentile: float, target_val: float = None, scale: float = 1):
    """
    Parameters
    ----------
    raw_img : np.ndarray
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


def enhance_contrast(img: np.ndarray, factor=5.0):
    im = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(im)
    im_output = enhancer.enhance(factor)
    return np.array(im_output)


def dilate_or_erode_mask(cropped_mask: np.ndarray, scale_factor):
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


def dilate_or_erode_label_mask(label_mask: np.ndarray, scale_factor, bg_val=0):
    """Erode label mask to make each labeled region smaller and thus separated."""
    labels = np.unique(label_mask)
    # remove bg label
    labels = labels[labels != bg_val]

    res = np.zeros(label_mask.shape)
    for label in labels:
        bin_mask = label_mask == label
        bin_mask = bin_mask.astype(np.uint8)
        eroded_mask = dilate_or_erode_mask(bin_mask, scale_factor=scale_factor)
        res = res + eroded_mask * label
    return res


def label_mask_to_edt_mask(label_mask, bg_val=0, dtype=np.uint8, normalize_func=normalize_img_to_uint8):
    labels = np.unique(label_mask)
    # remvoe bg_val
    labels = labels[labels != bg_val]
    edt_mask = np.zeros(label_mask.shape, dtype=np.float32)
    for label in labels:
        tmp_mask = label_mask == label
        # perform euclidean distance transform and normalize
        tmp_mask = ndimage.distance_transform_edt(tmp_mask)
        normalized_mask = normalize_func(tmp_mask)
        tmp_mask[tmp_mask != bg_val] = normalized_mask[tmp_mask != bg_val]
        edt_mask += tmp_mask

    # TODO: remove the guard below because it is unlikely that we will have a label mask with values > 255, but we should handle this case
    # The reason for "unlikely" is that the label mask is usually generated from a binary mask
    # And thus the normalization process in the loop above will ensure that the values are in the range [0, 255]
    if edt_mask.max() > 255 and dtype == np.uint8:
        edt_mask = normalize_img_to_uint8(edt_mask)

    return edt_mask.astype(dtype)


def normalize_edt(edt_img, edt_max=5):
    """
    Normalize the input Euclidean Distance Transform (EDT) image.

    Parameters:
    - edt_img (ndarray): The input EDT image.
    - edt_max (float): The maximum value to which the EDT image will be normalized. Default is 4.

    Returns:
    - normalized_edt_img (ndarray): The normalized EDT image.

    """

    max_val = edt_img.max()
    factor = max_val / (edt_max - 1)
    edt_pos_mask = edt_img >= 1
    if type(edt_img) == np.ndarray:
        res_img = edt_img.copy()
    else:
        # torch.Tensor case
        res_img = edt_img.clone()
    res_img[edt_pos_mask] = edt_img[edt_pos_mask] / factor + 1
    # edt_img[edt_pos_mask] = edt_img[edt_pos_mask] / factor + 1
    return res_img
