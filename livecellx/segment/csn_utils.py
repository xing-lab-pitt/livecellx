import numpy as np
from skimage.segmentation import find_boundaries


def make_weight_from_bin_map(masks, w0=10, sigma=5):
    """
    Generate the weight maps as specified in the UNet paper for a set of binary masks.

    This function creates weight maps that help to emphasize the borders between
    different instances in the masks, which is useful for training segmentation models
    like UNet. The border emphasis is controlled by parameters `w0` and `sigma`.

    Adapated from U-Net paper and implementation from Dr. Weikang Wang

    Parameters
    ----------
    masks : np.ndarray
        A 3D array of shape (n_masks, image_height, image_width),
        where each slice along the 0th axis represents one binary mask.
        Note that the masks are assumed to be binary, i.e., 0s and 1s.
    w0 : float, optional
        A weight factor for border emphasis, by default 10.
    sigma : float, optional
        The width of the Gaussian used to smooth the distance from the border,
        by default 5.

    Returns
    -------
    np.ndarray
        A 2D array of shape (image_height, image_width) representing the weight map.
    """
    n_masks, nrows, ncols = masks.shape
    masks = masks.astype(bool)
    dist_map = np.zeros((nrows * ncols, n_masks))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T

    for i, mask in enumerate(masks):
        # Find the boundary of each mask and compute the distance to the boundary for each pixel
        boundaries = find_boundaries(mask, mode="inner")
        X2, Y2 = np.nonzero(boundaries)
        dist_map[:, i] = np.sqrt(
            (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2 + (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        ).min(axis=0)

    # Compute the border loss map
    if n_masks == 1:
        d1 = dist_map.ravel()
        border_loss_map = w0 * np.exp(-(d1**2) / (2 * sigma**2))
    else:
        d1_ix, d2_ix = np.argpartition(dist_map, 2, axis=1)[:, :2].T
        d1 = dist_map[np.arange(dist_map.shape[0]), d1_ix]
        d2 = dist_map[np.arange(dist_map.shape[0]), d2_ix]
        border_loss_map = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma**2))

    # Reshape the border loss map to the original image dimensions
    border_loss_map_reshaped = np.zeros((nrows, ncols))
    border_loss_map_reshaped.ravel()[np.arange(nrows * ncols)] = border_loss_map

    # Compute class weight map
    w_1 = 1 - masks.sum() / (nrows * ncols)
    w_0 = 1 - w_1
    class_weight_map = np.where(masks.sum(axis=0), w_1, w_0)

    # Combine border loss and class weight maps
    return border_loss_map_reshaped + class_weight_map


def make_csn_weight_map(labels, w0=20, sigma=8):
    """
    Generate the weight maps for a set of label masks, where each label represents a different instance.

    This function is adapted to handle label masks directly, creating weight maps that help
    to emphasize the borders between different instances represented by unique labels in the masks.
    The border emphasis is controlled by parameters `w0` and `sigma`.

    Parameters
    ----------
    labels : np.ndarray
        A 2D array of shape (image_height, image_width), where each unique label
        represents a different instance.
    w0 : float, optional
        A weight factor for border emphasis, by default 10.
    sigma : float, optional
        The width of the Gaussian used to smooth the distance from the border,
        by default 5.

    Returns
    -------
    np.ndarray
        A 2D array of shape (image_height, image_width) representing the weight map.
    """
    nrows, ncols = labels.shape
    unique_labels = np.unique(labels)
    border_loss_map_reshaped = np.zeros((nrows, ncols))
    class_weight_map = np.zeros((nrows, ncols))

    for label in unique_labels:
        if label == 0:  # Background label, continue to next label
            continue
        binary_mask = labels == label
        boundaries = find_boundaries(binary_mask, mode="inner")
        X2, Y2 = np.nonzero(boundaries)
        X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")

        # Correct distance calculation with broadcasting
        dist_map = np.sqrt(
            (X2[:, np.newaxis, np.newaxis] - X1[np.newaxis, :, :]) ** 2
            + (Y2[:, np.newaxis, np.newaxis] - Y1[np.newaxis, :, :]) ** 2
        )
        dist_map = dist_map.min(axis=0)

        border_loss_map = w0 * np.exp(-(dist_map**2) / (2 * sigma**2))
        border_loss_map_reshaped += border_loss_map

    # Normalize border loss map for the number of labels to prevent overemphasis
    border_loss_map_reshaped /= len(unique_labels) - 1  # Exclude background

    # Compute class weights and normalize
    total_pixels = nrows * ncols
    for label in unique_labels:
        if label == 0:
            continue
        label_area = np.sum(labels == label)
        w_label = 1 - (label_area / total_pixels)
        class_weight_map[labels == label] += w_label

    # Normalize class weight map to keep the scale consistent
    class_weight_map /= len(unique_labels) - 1  # Exclude background

    return border_loss_map_reshaped + class_weight_map
