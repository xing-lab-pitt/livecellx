import cv2
import numpy as np


def get_bbox_from_regionprops(regions):
    """Get bounding box from regionprops.

    Parameters
    ----------
    regions : list of skimage.measure._regionprops._RegionProperties
        List of region properties.

    Returns
    -------
    list of tuple
        List of bounding boxes in [x1, y1, x2, y2] format (skimage)
    """
    return [region.bbox for region in regions]


def bbox_skimage_to_cv2_order(bboxes):
    """Convert bounding box from skimage order to cv2 order.

    Parameters
    ----------
    bboxes : list of tuple
        List of bounding boxes.

    Returns
    -------
    list of tuple
        List of bounding boxes.
    """
    return [(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in bboxes]


def gray_img_to_rgb(img):
    """Convert gray image to rgb image.

    Parameters
    ----------
    img : np.ndarray
        Gray image.

    Returns
    -------
    np.ndarray
        RGB image.
    """
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def get_cv2_bbox(label_mask: np.array):
    """generate cv2 style bounding box from label mask

    Parameters
    ----------
    label_mask :
      label mask: W x H np array with each pixel value indicating the label of objects (index 1, 2, 3, ...). Note that labels are not required to be consecutive..

    Returns
    -------
    _type_
        _description_
    """
    regions = measure.regionprops(label_mask)
    bboxes = get_bbox_from_regionprops(regions)
    bboxes_cv2 = bbox_skimage_to_cv2_order(bboxes)
    return bboxes_cv2
