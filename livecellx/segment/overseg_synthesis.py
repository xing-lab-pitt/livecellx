from copy import deepcopy
from typing import List, Tuple
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.morphology import local_maxima, h_maxima
from skimage.measure import regionprops, label
import scipy.stats
import random

import livecellx
from livecellx.core.single_cell import SingleCellStatic
from livecellx.trajectory.contour_utils import get_cellTool_contour_points, viz_contours
from livecellx.preprocess.utils import dilate_or_erode_label_mask


def get_line_pixels(pt1, pt2, thickness=2, max_x=float("inf"), max_y=float("inf")):
    """get all pixel coordinates between two points

    Parameters
    ----------
    pt1 : _type_
        _description_
    pt2 : _type_
        _description_
    thickness : int, optional
        _description_, by default 3
    max_x : _type_, optional
        _description_, by default float("inf")
    max_y : _type_, optional
        _description_, by default float("inf")

    Returns
    -------
    _type_
        _description_
    """

    def _get_line_pixels_xy(x1, y1, x2, y2, max_x, max_y):
        # get the line equation
        if x1 == x2:
            m = 0
        else:
            m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        # get the pixels
        pixels = set()

        def add_xy(x, y):
            if y < max_y and x < max_x and y >= 0 and x >= 0:
                pixels.add((x, y))
            # simple thickening
            sx, sy = x - thickness, y - thickness
            for nx in range(sx, sx + thickness * 2):
                for ny in range(sy, sy + thickness * 2):
                    # if nx ny in bound
                    if nx >= 0 and ny >= 0 and nx < max_x and ny < max_y:
                        pixels.add((nx, ny))

        # exclude x2 purposely to follow skimage bbox conventions
        for x in range(x1, x2):
            y = int(m * x + b)
            add_xy(x, y)
            y = int(m * x + b) + 1
            add_xy(x, y)
        return pixels

    x1, y1 = pt1
    x2, y2 = pt2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x1 > x2:
        pt1, pt2 = pt2, pt1
        x1, y1, x2, y2 = x2, y2, x1, y1
    pixel_set1 = _get_line_pixels_xy(x1, y1, x2, y2, max_x, max_y)

    if y1 > y2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    _pixel_set2 = _get_line_pixels_xy(y1, x1, y2, x2, max_y, max_x)
    pixel_set2 = set()
    for coord in _pixel_set2:
        pixel_set2.add((coord[1], coord[0]))
    return pixel_set1.union(pixel_set2)


def divide_single_cell_brute_force(sample_sc: SingleCellStatic, seg_crop, sampled_points=None):

    # get the contour points
    contour_points = sample_sc.get_contour_coords_on_img_crop()
    assert len(contour_points) >= 2, "need more than 2 contour points to divide a cell"
    if sampled_points is None:
        sampled_points = random.choices(contour_points, k=2)
        print("sampled_points:", sampled_points)

    pt1 = sampled_points[0]
    pt2 = sampled_points[1]

    # TODO remove the resample code below
    # # resample if the points are on the same vertical line
    # while pt1[0] == pt2[0] or pt1[1] == pt2[1]:
    #     sampled_points = random.choices(contour_points, k=2)
    #     pt1 = sampled_points[0]
    #     pt2 = sampled_points[1]

    line_pixels = np.array(list(get_line_pixels(pt1, pt2, max_x=seg_crop.shape[0], max_y=seg_crop.shape[1])), dtype=int)

    if len(line_pixels) == 0:
        print("[WARN] the sampled line pixels is empty, skipping division process...")
        return

    seg_crop[line_pixels[:, 0], line_pixels[:, 1]] = 0
    # get the center of mass of the contour
    center_of_mass = sample_sc

    # TODO: remove deepcopy because it creates underlying dataset objects holding paths as well
    new_sc = deepcopy(sample_sc)
    sample_sc.get_contour_img()


def add_random_gauss_to_img(
    contour_mask, raw_crop, square_len, gauss_center_val=200, gauss_std=8, inplace=False, pos=None
):
    # add random gaussian noise to the raw image
    # randomly choose a point inside the contour
    if not inplace:
        raw_crop = raw_crop.copy()
    np.where(contour_mask > 0)
    cell_points = np.where(contour_mask > 0)

    # # double check: viz cell points from np
    # temp = np.zeros(contour_mask.shape)
    # temp[cell_points] = 1
    # plt.imshow(temp)
    # plt.title("temp")
    # plt.show()
    if pos is not None:
        rand_pt = pos
    else:
        rand_idx = np.random.randint(0, len(cell_points[0]))
        rand_pt = np.array([cell_points[0][rand_idx], cell_points[1][rand_idx]])

    # the mesh grid is the square around the random point
    x_min, x_max = max(rand_pt[0] - square_len, 0), min(rand_pt[0] + square_len, contour_mask.shape[0])
    y_min, y_max = max(rand_pt[1] - square_len, 0), min(rand_pt[1] + square_len, contour_mask.shape[1])
    grid = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    grid = np.stack(grid, axis=-1)

    # compute the distance between the random point and the mesh grid
    dist_to_center = np.linalg.norm(grid - rand_pt, axis=-1)

    # calculate the gaussian cdf  based on dist to center
    gaussian_cdf = scipy.stats.norm.cdf(-dist_to_center, loc=0, scale=gauss_std)

    # add gaussian noise to seg_crop
    raw_crop[x_min:x_max, y_min:y_max] += gaussian_cdf.T * gauss_center_val

    # print("cell_points:", cell_points)
    # print("cell_points len:", len(cell_points[0]))
    # print("contour mask sum:", contour_mask.sum())
    # print("square_len:", square_len)
    # print("grid shape:", grid.shape)
    # print("dist_to_center shape:", dist_to_center.shape)
    # print("dist_to_center mean:", np.mean(dist_to_center))
    # print("gaussian noise shape:", gaussian_cdf.shape)
    # print("gaussian mean:", np.mean(gaussian_cdf))
    return raw_crop, rand_pt


def divide_single_cell_watershed(
    sample_sc: SingleCellStatic,
    raw_crop=None,
    peak_distance=20,
    markers=None,
    marker_method="hmax",
    h_threshold=1,
    normalize=True,
    normalize_edt=True,
    gauss_center_val=200,
    edt_gauss_center_val=1,
    gauss_std=8,
    num_gauss_areas=2,
    return_all=False,
):
    contour_points = sample_sc.get_contour_coords_on_img_crop()
    contour_mask = sample_sc.get_contour_mask()

    if raw_crop is None:
        raw_crop = sample_sc.get_contour_img()
    else:
        raw_crop = raw_crop.copy()

    # normalize seg_crop
    if normalize:
        raw_crop = livecellx.preprocess.utils.normalize_img_to_uint8(raw_crop)
        raw_crop = raw_crop.astype(float)

    # # print statistics of seg_crop
    # print("seg_crop shape:", raw_crop.shape)
    # print("seg_crop unique:", np.unique(raw_crop))
    # print("seg_crop mean:", np.mean(raw_crop))
    # print("cell area:", prop.area)
    # print("cell axis_major_length:", prop.major_axis_length)
    # print("cell axis_minor_length:", prop.minor_axis_length)

    # edt transform
    edt_distance = ndimage.distance_transform_edt(contour_mask)
    if normalize_edt:
        edt_flattened = edt_distance.flatten()
        edt_distance = (edt_distance - np.min(edt_flattened)) / (np.max(edt_flattened) - np.min(edt_flattened))
    # # print stats of edt_distance
    # print("edt_distance shape:", edt_distance.shape)
    # print("edt max:", np.max(edt_distance))
    # print("edt min:", np.min(edt_distance))
    # print("edt mean:", np.mean(edt_distance))
    # TODO: add gaussian noise

    # determin the area of noise (new labeled region for oversegmentation)
    assert len(np.unique(contour_mask)) == 2, "seg_crop should only contain one label"
    props = regionprops(label_image=contour_mask.astype(int), intensity_image=raw_crop)
    assert len(props) == 1, "seg_crop should only contain one label"
    prop = props[0]

    square_len = int(np.sqrt(prop.area))
    for _ in range(num_gauss_areas):
        _, rand_pos = add_random_gauss_to_img(
            contour_mask, raw_crop, square_len, gauss_center_val=gauss_center_val, gauss_std=gauss_std, inplace=True
        )
        add_random_gauss_to_img(
            contour_mask,
            edt_distance,
            square_len,
            gauss_center_val=edt_gauss_center_val,
            gauss_std=gauss_std,
            inplace=True,
            pos=rand_pos,
        )

    # # show edt_distance
    # plt.imshow(edt_distance)
    # plt.title("edt_distance")
    # plt.show()
    # watershed segmentation
    if markers is None and marker_method == "hmax":
        # local_hmax = h_maxima(raw_crop, h_threshold)
        local_hmax = h_maxima(edt_distance, h_threshold)
        markers = label(local_hmax, connectivity=1)
    elif markers is None and marker_method == "local":
        # use local peak as default markers
        coords = peak_local_max(edt_distance, min_distance=peak_distance, footprint=np.ones((3, 3)))
        mask = np.zeros(edt_distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)


def gen_synthetic_overseg(sc, num_samples=10, max_try=20, **kwargs) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    res_label_masks_and_params = []
    num_gauss_area = kwargs["num_gauss_areas"]
    for _ in range(num_samples):
        counter = 0
        num_segs = -1
        while num_segs < num_gauss_area and counter < max_try:
            label_mask = divide_single_cell_watershed(sc, **kwargs)
            num_segs = len(np.unique(label_mask)) - 1
            counter += 1
        if num_segs < num_gauss_area:
            # print("fail to generate enough segs")
            continue
        meta = kwargs.copy()
        meta["num_segs"] = num_segs
        eroded_label_mask = dilate_or_erode_label_mask(label_mask, scale_factor=-0.1, bg_val=0)
        res_label_masks_and_params.append((label_mask, eroded_label_mask, meta))
    return res_label_masks_and_params


def process_sc_synthetic_overseg_crops(
    sc, overseg_uns_key="overseg_imgs", num_samples=5, num_gauss_areas=np.arange(2, 6)
):
    sc.uns[overseg_uns_key] = []
    for num_gauss_area in num_gauss_areas:
        label_masks_and_params_hmax = gen_synthetic_overseg(
            sc,
            num_samples=num_samples,
            peak_distance=10,
            num_gauss_areas=num_gauss_area,
            marker_method="hmax",
            edt_gauss_center_val=10,
            gauss_std=16,
            h_threshold=1,
        )
        label_masks_and_params_local = gen_synthetic_overseg(
            sc,
            num_samples=num_samples,
            peak_distance=10,
            num_gauss_areas=num_gauss_area,
            edt_gauss_center_val=10,
            gauss_std=16,
            marker_method="local",
            gauss_center_val=150,
        )
        sc.uns[overseg_uns_key].extend(label_masks_and_params_hmax)
        sc.uns[overseg_uns_key].extend(label_masks_and_params_local)
    return sc
