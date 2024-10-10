import cv2
import numpy as np
from scipy import ndimage
from skimage import measure

from livecellx.preprocess.utils import normalize_img_to_uint8


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


def rgb_img_to_gray(img):
    """Convert rgb image to gray image.

    Parameters
    ----------
    img : np.ndarray
        RGB image.

    Returns
    -------
    np.ndarray
        Gray image.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def get_cv2_bbox(label_mask: np.ndarray):
    """generate cv2 style bounding box from a label mask

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


def clip_polygon(polygon, h, w):
    """
    The Sutherland-Hodgman algorithm. Adapted from ChatGpt's implementation.
    Define the Polygon and Clipping Rectangle: The polygon is defined by its vertices, and the clipping rectangle is defined by the dimensions H and W.
    Implement the Clipping Algorithm: The Sutherland-Hodgman algorithm is a common choice for polygon clipping. This algorithm iteratively clips the edges of the polygon against each edge of the clipping rectangle.
    Handle Edge Cases: Special care must be taken to handle edge cases, such as when a polygon vertex lies exactly on a clipping boundary.
    """

    def clip_edge(polygon, x1, y1, x2, y2):
        new_polygon = []
        for i in range(len(polygon)):
            current_x, current_y = polygon[i]
            previous_x, previous_y = polygon[i - 1]

            # Check if current and previous points are inside the clipping edge
            inside_current = (x2 - x1) * (current_y - y1) > (y2 - y1) * (current_x - x1)
            inside_previous = (x2 - x1) * (previous_y - y1) > (y2 - y1) * (previous_x - x1)

            if inside_current:
                if not inside_previous:
                    # Compute intersection and add to new polygon
                    new_polygon.append(intersect(previous_x, previous_y, current_x, current_y, x1, y1, x2, y2))
                new_polygon.append((current_x, current_y))
            elif inside_previous:
                # Compute intersection and add to new polygon
                new_polygon.append(intersect(previous_x, previous_y, current_x, current_y, x1, y1, x2, y2))

        return new_polygon

    def intersect(px, py, qx, qy, ax, ay, bx, by):
        # Compute the intersection point
        det = (qx - px) * (by - ay) - (qy - py) * (bx - ax)
        if det == 0:
            return qx, qy  # Lines are parallel
        l = ((by - ay) * (bx - px) - (bx - ax) * (by - py)) / det
        return px + l * (qx - px), py + l * (qy - py)

    # Clipping against four edges of the rectangle
    clipped_polygon = clip_edge(polygon, 0, 0, w, 0)
    clipped_polygon = clip_edge(clipped_polygon, w, 0, w, h)
    clipped_polygon = clip_edge(clipped_polygon, w, h, 0, h)
    clipped_polygon = clip_edge(clipped_polygon, 0, h, 0, 0)

    return np.array(clipped_polygon)


def crop_or_pad_img(img_crop, fix_dims):
    """Crop or pad a 2D image to fix_dims; For crop, crop from the central region"""
    if fix_dims is not None:
        if img_crop.shape[0] > fix_dims[0]:
            start = (img_crop.shape[0] - fix_dims[0]) // 2
            img_crop = img_crop[start : start + fix_dims[0], :]
        else:
            pad = (fix_dims[0] - img_crop.shape[0]) // 2
            img_crop = np.pad(img_crop, ((pad, pad), (0, 0)), mode="constant", constant_values=0)

        if img_crop.shape[1] > fix_dims[1]:
            start = (img_crop.shape[1] - fix_dims[1]) // 2
            img_crop = img_crop[:, start : start + fix_dims[1]]
        else:
            pad = (fix_dims[1] - img_crop.shape[1]) // 2
            img_crop = np.pad(img_crop, ((0, 0), (pad, pad)), mode="constant", constant_values=0)

    return img_crop
