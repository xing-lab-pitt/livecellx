from typing import Union
from typing import Dict
import mahotas.features.texture
import skimage
import skimage.measure
from pandas import Series
import numpy as np
import pandas as pd

from livecell_tracker.core.single_cell import SingleCellStatic


def compute_haralick_features(
    sc: SingleCellStatic, ignore_zeros=True, return_mean=True, ret_arr=True, **kwargs
) -> Union[np.array, Series]:
    """Returns a list of texture features for the given image.

    Parameters
    ----------
    image : ndarray
        The image to extract features from.

    Returns
    -------
    list
        A list of texture features.
    """
    image = sc.get_contour_img_crop()
    features = mahotas.features.texture.haralick(image, ignore_zeros=ignore_zeros, return_mean=return_mean, **kwargs)
    if ret_arr:
        return features
    feature_names = ["haralick_" + str(i) for i in range(len(features.flatten()))]
    return Series(features.ravel(), index=feature_names)


# https://scikit-image.org/docs/stable/api/skimage.measure.html
SELECTED_SKIMAGE_REGIONPROPOS_COL_DTYPES = {
    "area": float,
    "area_bbox": float,
    "area_convex": float,
    "area_filled": float,
    "axis_major_length": float,
    "axis_minor_length": float,
    "bbox": int,
    "centroid": float,
    "centroid_local": float,
    "centroid_weighted": float,
    "centroid_weighted_local": float,
    "eccentricity": float,
    "equivalent_diameter_area": float,
    "euler_number": int,
    "extent": float,
    "feret_diameter_max": float,
    "inertia_tensor": float,
    "inertia_tensor_eigvals": float,
    "intensity_max": float,
    "intensity_mean": float,
    "intensity_min": float,
    "label": int,
    "moments": float,
    "moments_central": float,
    "moments_hu": float,
    "moments_normalized": float,
    "moments_weighted": float,
    "moments_weighted_central": float,
    "moments_weighted_hu": float,
    "moments_weighted_normalized": float,
    "orientation": float,
    "perimeter": float,
    "perimeter_crofton": float,
    "solidity": float,
}


def compute_skimage_regionprops(
    sc: SingleCellStatic, props=SELECTED_SKIMAGE_REGIONPROPOS_COL_DTYPES.keys()
) -> pd.Series:
    label_mask = sc.get_contour_mask().astype(int)
    intensity_mask = sc.get_contour_img_crop()
    regionprops_results = skimage.measure.regionprops_table(label_mask, intensity_mask, properties=props)
    feature_keys = list(regionprops_results.keys())

    # Convert to pandas series
    for key in feature_keys:
        if not hasattr(regionprops_results[key], "__len__"):
            pass
        if len(regionprops_results[key]) == 1:
            regionprops_results[key] = regionprops_results[key][0]
        elif len(regionprops_results[key]) != 1:
            # TODO: Handle this case, probably by appending index suffix to key
            raise ValueError(
                "Regionprops should only return one value per property, %s contains %d values"
                % (key, len(regionprops_results[key]))
            )
    res_table = pd.Series(regionprops_results)
    return res_table
