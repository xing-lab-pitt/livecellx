import cv2
import cv2 as cv
import itertools
from skimage.measure import find_contours
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
from livecellx.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset
from livecellx.preprocess.utils import (
    overlay,
    enhance_contrast,
    normalize_img_to_uint8,
)
from livecellx.segment.ou_utils import csn_augment_helper, underseg_overlay_gt_masks, gen_aug_diff_mask


def viz_check_combined_sc_result(sc1, sc2):
    fig, axes = plt.subplots(1, 8, figsize=(18, 5))
    ax_idx = 0
    ax = axes[ax_idx]
    sc1.show_whole_img(ax=ax)
    ax.set_title("sc1 whole img")
    ax_idx += 1

    ax = axes[ax_idx]
    sc1.show(ax=ax)
    ax.set_title("sc1 img crop")
    ax_idx += 1

    ax = axes[ax_idx]
    sc1.show_mask(ax=ax, padding=20)
    ax.set_title("sc1 mask")
    ax_idx += 1

    ax = axes[ax_idx]
    sc1.show_contour_img(ax=ax, padding=20)
    ax.set_title("sc1 contour img")
    ax_idx += 1

    ax = axes[ax_idx]
    sc2.show_whole_img(ax=ax)
    ax.set_title("sc2 whole img")
    ax_idx += 1

    ax = axes[ax_idx]
    sc2.show(ax=ax, crop=True)
    ax.set_title("sc2 img crop")
    ax_idx += 1

    ax = axes[ax_idx]
    sc2.show_mask(ax=ax, padding=20)
    ax.set_title("sc2 mask")
    ax_idx += 1

    ax = axes[ax_idx]
    sc2.show_contour_img(ax=ax, padding=20)
    ax.set_title("sc2 contour img")
    ax_idx += 1

    plt.show()


def compute_distance_by_contour(sc1, sc2):
    # compute distance between two scs by their contour sample points
    c1, c2 = sc1.get_contour(), sc2.get_contour()
    return compute_two_contours_min_distance(c1, c2)


def check_contour_in_boundary(contour, boundary):
    return np.all(contour >= 0) and np.all(contour < boundary)


def adjust_contour_to_bounds(contour, bounds, bound_shift=-1):
    bounds = np.array(bounds)
    if not check_contour_in_boundary(contour, bounds):
        contour = contour.copy()
        contour[contour < 0] = 0
        contour = np.where(contour >= bounds, bounds + bound_shift, contour)
    return contour


def shift_contour_randomly(sc_center, contour, bounds):
    random_center = np.random.randint(low=0, high=bounds, size=2)
    shift = random_center - sc_center
    shift = shift.astype(int)
    contour_shifted = contour + shift
    return random_center, contour_shifted, shift


def compute_two_contours_min_distance(contour1, contour2):
    min_dist = np.inf
    for p1 in contour1:
        for p2 in contour2:
            dist = np.linalg.norm(p1 - p2)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def _add_sc_to_img_helper(sc, new_img, new_sc_mask, shift, in_place=False):
    if not in_place:
        new_img = new_img.copy()
    sc_ori_space_pixel_xy_arr = np.array(new_sc_mask.nonzero()).T - shift
    sc_ori_space_pixel_xy_arr[sc_ori_space_pixel_xy_arr < 0] = 0
    new_img[new_sc_mask] = sc.get_contour_img()[sc_ori_space_pixel_xy_arr[:, 0], sc_ori_space_pixel_xy_arr[:, 1]]
    return new_img


def add_sc_to_img(sc, new_img, mask, bg_img, in_place=False, mask_inplace=True, fix_sc_pos=False):
    bg_shape = np.array(bg_img.shape)
    sc_prop = sc.compute_regionprops()
    sc_contour_coords = sc.get_contour_coords_on_crop().astype(int)
    if fix_sc_pos:
        sc_new_center = sc_prop.centroid
        sc_new_contour = sc_contour_coords
        shift = 0
    else:
        sc_new_center, sc_new_contour, shift = shift_contour_randomly(
            sc_prop.centroid, sc_contour_coords, bounds=bg_shape
        )
    sc_new_contour = adjust_contour_to_bounds(sc_new_contour, bg_shape)
    new_sc_mask = SingleCellStatic.gen_contour_mask(sc_new_contour, bg_img, bbox=None, crop=False)
    new_sc_mask_bool = new_sc_mask > 0  # convert to bool
    new_img = _add_sc_to_img_helper(sc, new_img, new_sc_mask, shift, in_place=in_place)

    if mask_inplace:
        mask = mask.copy()
    mask[new_sc_mask_bool] = True
    return new_img, sc_new_contour, mask, shift


def combine_two_scs_monte_carlo(sc1, sc2, bg_img=None, bg_scale=1.5, fix_sc1=False, center_two_cells=False):
    def _gen_empty_bg_img():
        sc1_shape = sc1.get_img_crop().shape
        sc2_shape = sc2.get_img_crop().shape
        bg_shape = np.array([max(sc1_shape[0], sc2_shape[0]), max(sc1_shape[1], sc2_shape[1])])
        bg_shape = (bg_shape * bg_scale).astype(int)
        bg_img = np.zeros(shape=bg_shape)
        return bg_img

    if bg_img is None:
        bg_img = _gen_empty_bg_img()

    bg_shape = np.array(bg_img.shape)
    new_img = bg_img.copy()
    new_mask = np.zeros(shape=bg_shape, dtype=bool)

    _, sc1_new_contour, sc1_new_mask, shift1 = add_sc_to_img(
        sc1, new_img, bg_img=bg_img, mask=new_mask, in_place=True, fix_sc_pos=fix_sc1
    )
    _, sc2_new_contour, sc2_new_mask, shift2 = add_sc_to_img(sc2, new_img, bg_img=bg_img, mask=new_mask, in_place=True)

    new_sc1 = SingleCellStatic(
        timeframe=SingleImageDataset.DEFAULT_TIME,
        contour=sc1_new_contour,
        img_dataset=SingleImageDataset(new_img),
        mask_dataset=SingleImageDataset(sc1_new_mask),
    )
    new_sc2 = SingleCellStatic(
        timeframe=SingleImageDataset.DEFAULT_TIME,
        contour=sc2_new_contour,
        img_dataset=SingleImageDataset(new_img),
        mask_dataset=SingleImageDataset(sc2_new_mask),
    )
    if center_two_cells:
        new_sc1, new_scs = center_two_cells(new_sc1, bg_img, [new_sc2])
        new_sc2 = new_scs[0]
    return new_sc1, new_sc2, bg_img


def gen_synthetic_overlap_scs(
    sc1, sc2, max_overlap_percent=0.2, bg_scale=2.0, fix_sc1=False, min_reserved_area_percent=0.7, max_try=1000
):
    # TODO: optimize in the future via computational geometry; now simply use monte carlo for generating required synthetic data
    is_success = False
    counter = 0
    while not is_success and counter < max_try:
        is_success = True
        new_sc1, new_sc2, bg_img = combine_two_scs_monte_carlo(
            sc1, sc2, bg_img=None, bg_scale=bg_scale, fix_sc1=fix_sc1
        )
        # check overlap
        overlap_mask = np.logical_and(new_sc1.get_mask(), new_sc2.get_mask())
        overlap_percent = float(np.sum(overlap_mask)) / min(np.sum(new_sc1.get_mask()), np.sum(new_sc2.get_mask()))
        if overlap_percent > 0 and overlap_percent < max_overlap_percent:
            pass
        else:
            is_success = False

        # check area percent to prevent scs that are too small
        area = float(np.sum((new_sc1.get_contour_mask() > 0).flatten())) + np.sum(
            (new_sc2.get_contour_mask() > 0).flatten()
        )
        old_area = float(np.sum((sc1.get_contour_mask() > 0).flatten())) + np.sum(
            (sc2.get_contour_mask() > 0).flatten()
        )
        if (area / old_area) < min_reserved_area_percent:
            is_success = False
        counter += 1

    # return new_sc1, new_sc2, overlap_percent, is_success
    return {
        "new_sc1": new_sc1,
        "new_sc2": new_sc2,
        "overlap_percent": overlap_percent,
        "is_success": is_success,
        "bg_img": bg_img,
    }


def gen_gauss_sc_bg(sc: SingleCellStatic, shape):
    """generate background for sc by gaussian noise"""
    img = sc.get_img()
    mask = sc.get_mask().astype(bool)
    bg_mask = np.logical_not(mask)

    # compute gauss distribution of background pixels
    pixels = img[bg_mask].flatten()
    mean = np.mean(pixels)
    std = np.std(pixels)
    res_bg_img = np.random.normal(0, 1, shape)
    res_bg_img = res_bg_img * std + mean
    return res_bg_img


def gen_sc_bg_crop(sc, shape):
    """generate background for sc by cropping from the sc's image. For regions belong to single cells, remove cells and fill with gaussian noise."""
    img = sc.get_img()
    mask = sc.get_mask().astype(bool)
    bg_mask = np.logical_not(mask)
    if not (img.shape[0] >= shape[0] and img.shape[1] >= shape[1]):
        print("Shape of sc is smaller than the required shape, return None...")
        return None
    # compute gauss distribution of background pixels
    pixels = img[bg_mask].flatten()
    mean = np.mean(pixels)
    std = np.std(pixels)

    # randomly crop a region from the large image
    bounds = np.array(img.shape) - np.array(shape)
    crop_row, crop_col = np.random.randint(low=0, high=bounds, size=2)
    res_bg_img = np.array(img[crop_row : crop_row + shape[0], crop_col : crop_col + shape[1]])
    res_bg_mask = np.array(bg_mask[crop_row : crop_row + shape[0], crop_col : crop_col + shape[1]])

    res_bg_img[~res_bg_mask] = np.random.normal(0, 1, np.sum(~res_bg_mask)) * std + mean
    return res_bg_img


def move_two_scs(sc1: SingleCellStatic, sc2: SingleCellStatic, pos_offset_vec, sc1_ori_img, inplace=False):
    if not inplace:
        sc1 = sc1.copy()
        sc2 = sc2.copy()
    img_space_dims = sc1_ori_img.shape
    pos_offset_vec = np.array(pos_offset_vec).astype(int)
    new_contour = np.array(sc2.get_contour()) + pos_offset_vec
    contour_before_adjust = new_contour.copy()
    tmp_sc2_bbox_before_adjust = SingleCellStatic.get_bbox_from_contour(contour_before_adjust)
    new_contour = adjust_contour_to_bounds(new_contour, img_space_dims)
    tmp_sc2 = sc2.copy()
    tmp_sc2.update_contour(new_contour, update_bbox=True)

    new_img = sc1_ori_img.copy()
    new_mask = np.zeros(sc1_ori_img.shape, dtype=np.uint8)
    sc1_bbox, sc2_bbox = sc1.get_bbox(), sc2.get_bbox()
    new_sc_bbox = tmp_sc2.get_bbox()
    projected_new_sc_bbox = (new_sc_bbox.reshape(2, 2) - pos_offset_vec).flatten()

    # Note that the boundaries are imgage dims + 1 because if skimage bbox's definition is [min_row, min_col, max_row, max_col)
    projected_new_sc_bbox = adjust_contour_to_bounds(
        projected_new_sc_bbox.reshape(2, 2), np.array(img_space_dims), bound_shift=0
    ).flatten()
    if projected_new_sc_bbox[0] == img_space_dims[0]:
        projected_new_sc_bbox[0] -= 1
    if projected_new_sc_bbox[1] == img_space_dims[1]:
        projected_new_sc_bbox[1] -= 1

    # fix a corner case that may cause the projected_new_sc_bbox to be empty
    # if projected_new_sc_bbox[2] == projected_new_sc_bbox[0]:
    #     projected_new_sc_bbox[2] += 1
    # if projected_new_sc_bbox[3] == projected_new_sc_bbox[1]:
    #     projected_new_sc_bbox[3] += 1
    # print("dims: ", img_space_dims)
    # print("projected_new_sc_bbox: ", projected_new_sc_bbox)
    # print("new_sc_bbox: ", new_sc_bbox)

    # update datasets
    # TODO: consider if we have more datasets in single cell objects?
    sc1_contour_mask = sc1.get_contour_mask()
    sc2_contour_mask__projected = sc2.get_contour_mask(bbox=projected_new_sc_bbox)

    # Note: we do not need to set sc1 image here because it is included in sc1_ori_img
    # The reason for using sc1_ori_img is that when moving cells apart, sc2's part may remain in sc1 and we need to keep the original image of sc1

    # new_img[sc1_bbox[0] : sc1_bbox[2], sc1_bbox[1] : sc1_bbox[3]][sc1_contour_mask] = sc1.get_contour_img()[
    #     sc1_contour_mask
    # ]

    new_img[new_sc_bbox[0] : new_sc_bbox[2], new_sc_bbox[1] : new_sc_bbox[3]][
        sc2_contour_mask__projected
    ] = sc2.get_contour_img(bbox=projected_new_sc_bbox)[sc2_contour_mask__projected]

    new_mask[sc1_bbox[0] : sc1_bbox[2], sc1_bbox[1] : sc1_bbox[3]] |= sc1.get_contour_mask()
    new_mask[new_sc_bbox[0] : new_sc_bbox[2], new_sc_bbox[1] : new_sc_bbox[3]][
        sc2_contour_mask__projected
    ] |= sc2.get_mask_crop(bbox=projected_new_sc_bbox)[sc2_contour_mask__projected]

    # set image datasets of scs
    sc1.img_dataset = SingleImageDataset(new_img)
    sc1.mask_dataset = SingleImageDataset(new_mask)

    sc2.img_dataset = sc1.img_dataset
    sc2.mask_dataset = sc1.mask_dataset
    sc2.update_contour(tmp_sc2.get_contour(), update_bbox=True)

    return sc1, sc2


def move_two_syn_scs_close_or_apart(
    sc1: SingleCellStatic, sc2: SingleCellStatic, dist, bg_img, inplace=False, apart=False
):
    if not inplace:
        sc1 = sc1.copy()
        sc2 = sc2.copy()
    overlap = sc1.compute_overlap_percent(sc2)
    img_space_dims = bg_img.shape
    assert overlap <= 1e-5, "Two scs should not overlap"

    # move sc2 toward sc1
    norm_vec = sc1.get_center(crop=False) - sc2.get_center(crop=False)
    norm_vec = norm_vec / np.linalg.norm(norm_vec)
    pos_offset_vec = (norm_vec * dist).astype(int)
    if apart:
        pos_offset_vec = -pos_offset_vec
    return move_two_scs(sc1, sc2, pos_offset_vec=pos_offset_vec, sc1_ori_img=bg_img)


def move_util_in_range(
    sc1: SingleCellStatic,
    sc2: SingleCellStatic,
    dist_per_move,
    bg_img,
    min_dist=-np.inf,
    max_dist=np.inf,
    inplace=False,
    max_move=100,
    allow_overlap=False,
):
    """utilize move_two_syn_scs_close_or_apart to move two scs within a certain distance range"""
    if not inplace:
        sc1 = sc1.copy()
        sc2 = sc2.copy()

    # move sc2 toward sc1
    norm_vec = sc1.get_center(crop=False) - sc2.get_center(crop=False)
    norm_vec = norm_vec / np.linalg.norm(norm_vec)

    cur_dist = compute_two_contours_min_distance(sc1.get_contour(), sc2.get_contour())
    if dist_per_move is None:
        # # TODO: make distance per move more efficient
        # if cur_dist > max_dist:
        #     dist_per_move = cur_dist / 2
        # elif cur_dist < min_dist:
        #     dist_per_move = cur_dist / 2
        # else:
        #     dist_per_move = (max_dist - min_dist) / 2
        dist_per_move = cur_dist / 2
        # dist_per_move = (max_dist - min_dist) / 2

    pos_offset_vec_toward = (norm_vec * dist_per_move).astype(int)
    pos_offset_vec = pos_offset_vec_toward
    if cur_dist < min_dist:
        pos_offset_vec = -pos_offset_vec

    # print("start dist: ", cur_dist, "pos_offset_vec: ", pos_offset_vec, "min_dist: ", min_dist, "max_dist: ", max_dist, "allow_overlap: ", allow_overlap)
    counter = 0
    iou = sc1.compute_iou(sc2)

    # when moving sc2, we may overwrite some part of sc1
    sc1_ori_img = sc1.get_img()
    while (cur_dist < min_dist or cur_dist > max_dist or (not allow_overlap and iou > 0)) and counter < max_move:
        sc1, sc2 = move_two_scs(sc1, sc2, pos_offset_vec=pos_offset_vec, sc1_ori_img=sc1_ori_img, inplace=inplace)
        cur_dist = compute_two_contours_min_distance(sc1.get_contour(), sc2.get_contour())
        iou = sc1.compute_iou(sc2)

        if cur_dist < 2 and cur_dist < min_dist:
            dist_per_move = (max_dist - min_dist) / 2
        else:
            dist_per_move = cur_dist / 2

        norm_vec = sc1.get_center(crop=False) - sc2.get_center(crop=False)
        norm_vec = norm_vec / np.linalg.norm(norm_vec)
        pos_offset_vec_toward = (norm_vec * dist_per_move).astype(int)
        if iou > 0 or cur_dist <= min_dist:
            pos_offset_vec = -pos_offset_vec_toward
        else:
            pos_offset_vec = pos_offset_vec_toward

        counter += 1
        # if counter % 5 == 0:
        #     print("counter: ", counter, "cur_dist: ", cur_dist, "iou: ", iou, "pos_offset_vec: ", pos_offset_vec)
    return sc1, sc2


def gen_synthetic_nonoverlap_by_two_scs(
    sc1: SingleCellStatic,
    sc2: SingleCellStatic,
    min_dist=-np.inf,
    max_dist=np.inf,
    min_reserved_area_percent=0.9,
    bg_scale=3.0,
    fix_sc1=False,
    max_try=1000,
    gen_bg_func=None,
    use_move_close=True,
    dist_per_move=None,
):
    is_success = False
    counter = 0
    sc1_shape = sc1.get_contour_mask().shape
    sc2_shape = sc2.get_contour_mask().shape
    max_shape = np.max(np.array([sc1_shape, sc2_shape]), axis=0) * bg_scale

    syn_bg_shape = (int(max_shape[0]), int(max_shape[1]))
    while not is_success and counter < max_try:
        is_success = True
        bg_img = None
        if gen_bg_func is not None:
            bg_img = gen_bg_func(sc1, shape=syn_bg_shape)
            if bg_img is None:
                is_success = False
                continue

        # TODO: we can improve this function by replacing monte carlo method
        # TODO: calculate distance between two scs and move two scs together can satisfy the conditions efficiently
        new_sc1, new_sc2, bg_img = combine_two_scs_monte_carlo(
            sc1, sc2, bg_img=bg_img, bg_scale=bg_scale, fix_sc1=fix_sc1
        )
        if use_move_close:
            new_sc1, new_sc2 = move_util_in_range(
                new_sc1,
                new_sc2,
                dist_per_move=dist_per_move,
                bg_img=bg_img,
                min_dist=min_dist,
                max_dist=max_dist,
                inplace=True,
                max_move=100,
            )
        # check overlap
        overlap_percent = new_sc1.compute_iou(new_sc2)
        if overlap_percent > 1e-5:
            is_success = False
        dist = compute_distance_by_contour(new_sc1, new_sc2)
        if dist < min_dist or dist > max_dist:
            is_success = False

        # check area percent to prevent scs that are too small
        area = float(np.sum((new_sc1.get_contour_mask() > 0).flatten())) + np.sum(
            (new_sc2.get_contour_mask() > 0).flatten()
        )
        old_area = float(np.sum((sc1.get_contour_mask() > 0).flatten())) + np.sum(
            (sc2.get_contour_mask() > 0).flatten()
        )
        if (area / old_area) < min_reserved_area_percent:
            is_success = False

        # print("counter: {}, overlap: {}, dist: {}, %%area: {}".format(counter, overlap_percent, dist, area / old_area))
        counter += 1
    # return new_sc1, new_sc2, dist, is_success
    return {
        "new_sc1": new_sc1,
        "new_sc2": new_sc2,
        "dist": dist,
        "is_success": is_success,
        "bg_img": bg_img,
    }


def show_cv2_contours(contours, img):
    im = np.expand_dims(img.astype(np.uint8), axis=2).repeat(3, axis=2)
    for k, _ in enumerate(contours):
        im = cv.drawContours(im, contours, k, (0, 230, 255), 6)
    plt.imshow(im)
    plt.show()


def find_contours_opencv(mask) -> list:
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    for i, contour in enumerate(contours):
        contour = np.array(contour)
        contour = contour[:, :, ::-1]
        contour = contour.reshape(-1, 2)
        contours[i] = contour
    return contours


def find_label_mask_contours(label_mask, bg_val=0) -> list:
    labels = np.unique(label_mask)
    labels = labels[labels != bg_val]
    contours = []
    for label in labels:
        mask = label_mask == label
        contours += find_contours_opencv(mask)
    return contours


def merge_two_scs_overlap(sc1: SingleCellStatic, sc2: SingleCellStatic):
    new_mask = np.logical_or(sc1.get_mask().astype(bool), sc2.get_mask().astype(bool))
    # plt.imshow(new_mask)
    # plt.show()
    # print(np.unique(new_mask))
    # contours = find_contours(new_mask, fully_connected="high")

    contours = find_contours_opencv(new_mask.astype(np.uint8))
    assert len(contours) != 0, "must contain at least one contour"
    if len(contours) > 1:
        print("WARNING: more than one contour found, return merge failure to the caller...")
        return None, False
    new_contour = contours[0]
    res_sc = SingleCellStatic(
        timeframe=SingleImageDataset.DEFAULT_TIME,
        contour=new_contour,
        img_dataset=sc1.img_dataset,
        mask_dataset=SingleImageDataset(new_mask),
    )
    return res_sc, True


def merge_two_scs_nonoverlap(sc1: SingleCellStatic, sc2: SingleCellStatic, max_dilate_iter=998, kernel_shape=(4, 4)):
    new_mask = np.logical_or(sc1.get_mask().astype(bool), sc2.get_mask().astype(bool))

    contours = find_contours_opencv(new_mask.astype(np.uint8))
    assert len(contours) != 0, "must contain at least one contour"
    if len(contours) != 2:
        print(
            "WARNING: #contours should be exactly 2 for merging two cells in non-overlap case, return merge failure to the caller..."
        )
        return None, False

    # dilate until the two contours are merged
    kernel = np.ones(kernel_shape, np.uint8)
    counter = 0
    # TODO: optimize the loop (possibly by binary search for the optimal iteration)
    while len(contours) != 1 and counter < max_dilate_iter:
        new_mask = cv2.dilate(new_mask.astype(np.uint8), kernel, iterations=1)
        contours = find_contours_opencv(new_mask.astype(np.uint8))
        counter += 1

    if len(contours) != 1:
        print(
            "WARNING: #contours should be exactly 1 after merging two cells in the non-overlap case, return merge failure to the caller..."
        )
        return None, False

    new_contour = contours[0]
    res_sc = SingleCellStatic(
        timeframe=SingleImageDataset.DEFAULT_TIME,
        contour=new_contour,
        img_dataset=sc1.img_dataset,
        mask_dataset=SingleImageDataset(new_mask),
    )
    return res_sc, True


def augment_and_save_merged_sc(sc: SingleCellStatic, scale_factors, scs, img_id, seg_label, syn_id, out_dir):
    # img_crop = sc.get_img_crop()
    # seg_crop = sc.get_mask_crop()

    # Note: do not use get_img_crop() and get_mask_crop() here, because we want to use the original image and mask
    # which are consistent across scs passed in
    img_crop = sc.get_img()
    seg_crop = sc.get_mask()

    syn_underseg_out_dir = out_dir
    raw_out_dir = syn_underseg_out_dir / "raw"
    seg_out_dir = syn_underseg_out_dir / "seg"
    gt_out_dir = syn_underseg_out_dir / "gt"
    gt_label_out_dir = syn_underseg_out_dir / "gt_label_mask"
    augmented_seg_dir = syn_underseg_out_dir / "augmented_seg"
    raw_transformed_img_dir = syn_underseg_out_dir / "raw_transformed_img"
    augmented_diff_seg_dir = syn_underseg_out_dir / "augmented_diff_seg"

    # makedirs
    all_dirs = [
        syn_underseg_out_dir,
        raw_out_dir,
        seg_out_dir,
        gt_out_dir,
        gt_label_out_dir,
        augmented_seg_dir,
        raw_transformed_img_dir,
        augmented_diff_seg_dir,
    ]
    for directory in all_dirs:
        if not directory.exists():
            print(">>> creating dir: ", directory)
            os.makedirs(directory, exist_ok=True)

    # generate combined gt label mask
    combined_gt_label_mask = np.zeros(seg_crop.shape, dtype=int)
    for i, tmp_sc in enumerate(scs):
        # mask = SingleCellStatic.gen_skimage_bbox_img_crop(sc.bbox, tmp_sc.get_mask())
        mask = tmp_sc.get_mask().astype(bool)
        if mask.shape != combined_gt_label_mask.shape:
            print("mask dim: ", mask.shape)
            print("combined_gt_label_mask dim: ", combined_gt_label_mask.shape)
        combined_gt_label_mask[mask] = i + 1

    raw_img_path = raw_out_dir / ("syn-underseg-img-%d_seg-%d.tif" % (img_id, seg_label))
    seg_img_path = seg_out_dir / ("syn-underseg-img-%d_seg-%d.tif" % (img_id, seg_label))
    gt_img_path = gt_out_dir / ("syn-underseg-img-%d_seg-%d.tif" % (img_id, seg_label))
    gt_label_img_path = gt_label_out_dir / ("img-%d_seg-%d.tif" % (img_id, seg_label))

    underseg_train_tuples = []
    augmented_data = []
    filename_pattern = "syn-underseg-img-%d_seg-%d.tif"
    res_dict = csn_augment_helper(
        img_crop=img_crop,
        seg_label_crop=seg_crop,
        combined_gt_label_mask=combined_gt_label_mask,
        overseg_raw_seg_crop=None,
        overseg_raw_seg_img_path=None,
        scale_factors=scale_factors,
        train_path_tuples=underseg_train_tuples,
        augmented_data=augmented_data,
        img_id=img_id,
        seg_label=syn_id,
        gt_label=-1,
        raw_img_path=raw_img_path,
        seg_img_path=seg_img_path,
        gt_img_path=gt_img_path,
        gt_label_img_path=gt_label_img_path,
        augmented_seg_dir=augmented_seg_dir,
        augmented_diff_seg_dir=augmented_diff_seg_dir,
        filename_pattern=filename_pattern,
        raw_transformed_img_dir=raw_transformed_img_dir,
        df_save_path=syn_underseg_out_dir / "data.csv",
    )
    return res_dict


def center_scs(cur_merged_sc, bg_img, sc_comps=[], viz=False):
    merged_sc_img = cur_merged_sc.get_img()
    print("bg_img shape: ", bg_img.shape, "merged_sc_img shape: ", merged_sc_img.shape)

    merged_sc_bbox = cur_merged_sc.bbox
    merged_sc_mask = cur_merged_sc.get_mask_crop().astype(bool)
    # shift bbox to the center of the image
    m_bbox_height = merged_sc_bbox[2] - merged_sc_bbox[0]
    m_bbox_width = merged_sc_bbox[3] - merged_sc_bbox[1]
    bg_center = (bg_img.shape[0] // 2, bg_img.shape[1] // 2)
    center_start = (bg_center[0] - m_bbox_height // 2, bg_center[1] - m_bbox_width // 2)
    center_bbox = (center_start[0], center_start[1], center_start[0] + m_bbox_height, center_start[1] + m_bbox_width)

    center_shift = (center_bbox[0] - merged_sc_bbox[0], center_bbox[1] - merged_sc_bbox[1])

    # create a new image
    new_img = bg_img.copy()
    new_img[center_bbox[0] : center_bbox[2], center_bbox[1] : center_bbox[3]][merged_sc_mask] = merged_sc_img[
        merged_sc_bbox[0] : merged_sc_bbox[2], merged_sc_bbox[1] : merged_sc_bbox[3]
    ][merged_sc_mask]
    new_img_dataset = SingleImageDataset(new_img)

    # create a new mask
    new_mask = np.zeros(bg_img.shape, dtype=bool)
    new_mask[center_bbox[0] : center_bbox[2], center_bbox[1] : center_bbox[3]][merged_sc_mask] = True
    new_mask_dataset = SingleImageDataset(new_mask)

    # copy and update the contour
    new_contour = cur_merged_sc.contour.copy()
    new_contour[:, 0] += center_shift[0]
    new_contour[:, 1] += center_shift[1]

    res_merged_sc = cur_merged_sc.copy()
    res_merged_sc.update_contour(new_contour)
    res_merged_sc.img_dataset = new_img_dataset
    res_merged_sc.mask_dataset = new_mask_dataset

    if viz:
        print(">" * 20, "vizualizing merged cells", "<" * 20)
        print("center_bbox: ", center_bbox)
        print("center_shift: ", center_shift)
        print("merged_sc_bbox: ", merged_sc_bbox)
        # show old and new images
        fig, axes = plt.subplots(1, 4, figsize=(20, 10))
        axes[0].imshow(merged_sc_img)
        axes[0].set_title("original merged sc")
        axes[1].imshow(new_img)
        axes[1].set_title("new merged sc")
        axes[2].imshow(merged_sc_mask)
        axes[2].set_title("original merged sc mask")
        axes[3].imshow(new_mask)
        axes[3].set_title("new merged sc mask")

    res_scs = []
    for sc in sc_comps:
        sc_contour = sc.contour.copy()
        sc_contour[:, 0] += center_shift[0]
        sc_contour[:, 1] += center_shift[1]
        sc_bbox = sc.bbox
        new_sc = sc.copy()
        new_sc.update_contour(sc_contour)
        new_sc.img_dataset = new_img_dataset

        # TODO: optimize: if the sc masks are the same as the merged sc mask, we may reuse the mask dataset generated for the merged sc
        # we need a sc specific mask because the mask of the merged sc is possibly changed
        tmp_new_sc_mask = np.zeros(bg_img.shape, dtype=bool)
        tmp_new_sc_mask[new_sc.bbox[0] : new_sc.bbox[2], new_sc.bbox[1] : new_sc.bbox[3]] = sc.get_mask_crop().astype(
            bool
        )
        new_sc.mask_dataset = SingleImageDataset(tmp_new_sc_mask)

        res_scs.append(new_sc)
        if viz:
            print(">" * 20, "single cell component in merged cell", "<" * 20)
            new_sc.show_panel(padding=20)
    return res_merged_sc, res_scs


def gen_underseg_scs_sample(
    scs,
    num_cells,
    save_dir=None,
    sample_id=None,
    augment_scale_factors=None,
    viz_check=False,
    viz_padding=20,
    sc_generator_func=gen_synthetic_overlap_scs,
    sc_generator_func_kwargs={},
    merge_func=merge_two_scs_overlap,
    center=True,
):
    assert len(scs) > 0, "tmp_scs is empty"
    cur_merged_sc = scs[0].copy()
    # merged_scs contains each individual single AFTER merging
    _merged_syn_scs = None
    is_success = True
    is_gen_success = None
    is_merge_success = None

    for j in range(1, num_cells):
        is_success = True
        cur_sc = scs[j]
        res_dict = sc_generator_func(cur_merged_sc, cur_sc, fix_sc1=True, **sc_generator_func_kwargs)
        cur_merged_sc, new_sc2, is_gen_success = res_dict["new_sc1"], res_dict["new_sc2"], res_dict["is_success"]
        bg_img = res_dict["bg_img"]
        is_success &= is_gen_success
        if not is_success:
            break

        if _merged_syn_scs is None:
            _merged_syn_scs = [cur_merged_sc]

        _merged_syn_scs.append(new_sc2)

        assert (
            cur_merged_sc.get_mask().shape == new_sc2.get_mask().shape
        ), "contact developer: two generated underseg scs should have the same shape"
        cur_merged_sc, is_merge_success = merge_func(cur_merged_sc, new_sc2)
        is_success &= is_merge_success
        if not is_success:
            break

    # at some point, the merging process failed
    if not is_success:
        print("gen success:", is_gen_success)
        print("merge success:", is_merge_success)
        print("synthesize failure for combination:", scs)
        return {"is_success": False}
    # Now we make sure that the masks of the merged scs have the same shape (in the same space)
    # for operate them easier later (e.g. merge the mask and generate label masks)
    # the scs' bbox coordinates keeps the same, relative to the cur_merged_sc
    # the image and mask dataset may be different in each iteration above
    # thus we need to update the img_dataset and mask_dataset for each sc in merged_syn_scs
    for sc in _merged_syn_scs:
        sc.img_dataset = cur_merged_sc.img_dataset

        # all datasets below should be single image datasets
        cur_merged_mask_shape = cur_merged_sc.get_mask().shape
        sc_mask_in_merged_space = np.zeros(cur_merged_mask_shape, dtype=np.uint8)
        sc.update_bbox()

        # update sc_mask by intersection of bbox
        # note that the new sc_mask should be smaller than the original sc_mask due to our simulator's scale factor setting
        # sometimes the bbox is out of range, we need to check it.
        # TODO: investigate why the following condition is not always true...
        is_bbox_make_sense = (
            sc.bbox[0] >= 0
            and sc.bbox[1] >= 0
            and sc.bbox[2] <= cur_merged_mask_shape[0]
            and sc.bbox[3] <= cur_merged_mask_shape[1]
        )
        if not is_bbox_make_sense:
            print(
                "generated sc_mask of the merged cell is smaller than that of one sc: sc bbox is out of range, sc.bbox=%s, sc_mask_shape=%s"
                % (str(sc.bbox), str(cur_merged_mask_shape))
            )
            return {"is_success": False}

        # when we generate the synthetic underseg scs, we fix coordinates of the first sc, so the cooridnates of the synthetic cells are always fixed relative to the first sc and in the same space.
        sc_mask = sc.get_contour_mask().astype(bool)
        sc_mask_in_merged_space[sc.bbox[0] : sc.bbox[2], sc.bbox[1] : sc.bbox[3]][sc_mask] = 1
        # sc_mask_in_merged_space[sc.bbox[0] : sc.bbox[2], sc.bbox[1] : sc.bbox[3]] = sc.get_mask()[
        #     sc.bbox[0] : sc.bbox[2], sc.bbox[1] : sc.bbox[3]
        # ]
        if viz_check:
            # fig, axes = plt.subplots(1, 2)
            # axes[0].imshow(sc.get_mask())
            # axes[0].set_title("sc mask")
            # axes[1].imshow(sc_mask_in_merged_space)
            # axes[1].set_title("sc mask after update")
            # plt.show()
            pass
        sc.mask_dataset = SingleImageDataset(sc_mask_in_merged_space)
        contours = find_contours_opencv(sc_mask_in_merged_space)
        if len(contours) != 1:
            print(
                "[WARNING] #contours:",
                len(contours),
                " (!=1) when aligning synthetic single cells. Probably there is something wrong with contour finding algorithm we use. Discarding the current sample anyway...",
            )
            is_success = False
            return {
                "is_success": False,
            }
        sc.update_contour(contours[0])
        assert (
            sc.get_mask().shape == cur_merged_sc.get_mask().shape
        ), "Two generated underseg scs should have the same shape."
    if center:
        cur_merged_sc, _merged_syn_scs = center_scs(cur_merged_sc, bg_img, _merged_syn_scs)
    cur_merged_sc.meta = {
        "num_merged_cells": num_cells,
    }

    if viz_check:
        cur_merged_sc.show_panel(padding=viz_padding)

    df = None  # df is only generated when save_dir is provided
    if save_dir:
        assert sample_id is not None, "sample_id should be provided if save_dir is provided"
        assert augment_scale_factors is not None, "augment_scale_factors should be provided if save_dir is provided"
        res_dict = augment_and_save_merged_sc(
            cur_merged_sc,
            augment_scale_factors,
            scs=_merged_syn_scs,
            img_id=sample_id,
            seg_label=sample_id,
            syn_id=sample_id,
            out_dir=save_dir,
        )
        df = res_dict["df"]
    return {
        "is_success": is_success,
        "merged_scs": _merged_syn_scs,
        "cur_merged_sc": cur_merged_sc,
        "df": df,
    }


def _gen_underseg_scs_sample_wrapper(input_args):
    return gen_underseg_scs_sample(**input_args)


def gen_underseg_scs(
    scs,
    num_cells=3,
    total_sample_num=1000,
    return_scs=False,
    save_dir: Path = None,
    augment_scale_factors=np.linspace(0, 0.1, 10),
    shuffle=True,
    sample_id_offset=0,
    viz_check=False,
    parallel=True,
    sc_generator_func=gen_synthetic_overlap_scs,
    sc_generator_func_kwargs=dict(),
    merge_func=merge_two_scs_overlap,
):
    import random
    import math
    import tqdm

    scs = list(scs)
    # random.shuffle(scs)
    def _process_sequential():
        with tqdm.tqdm(total=total_sample_num) as pbar:
            counter = 0
            for i in range(num_cells):
                # sample n scs from scs
                tmp_scs = random.sample(scs, num_cells)
                sample_id = i + sample_id_offset
                res_data = gen_underseg_scs_sample(
                    tmp_scs,
                    num_cells,
                    viz_check=viz_check,
                    save_dir=save_dir,
                    sample_id=sample_id,
                    augment_scale_factors=augment_scale_factors,
                    sc_generator_func=sc_generator_func,
                    sc_generator_func_kwargs=sc_generator_func_kwargs,
                    merge_func=merge_func,
                )
                if not res_data["is_success"]:
                    continue
                _merged_scs = res_data["merged_scs"]
                cur_merged_sc = res_data["cur_merged_sc"]

                sample_id += 1
                counter += 1
                pbar.update(1)
                if counter >= total_sample_num:
                    break

    # parallel version
    def process_parallel(required_sample_num, all_df, counter, cur_id):
        inputs = []
        for i in range(required_sample_num):
            tmp_scs = random.sample(scs, num_cells)
            inputs.append(
                {
                    "scs": tmp_scs,
                    "num_cells": num_cells,
                    "save_dir": save_dir,
                    "sample_id": cur_id + sample_id_offset,
                    "augment_scale_factors": augment_scale_factors,
                    "viz_check": viz_check,
                    "sc_generator_func": sc_generator_func,
                    "sc_generator_func_kwargs": sc_generator_func_kwargs,
                    "merge_func": merge_func,
                }
            )
            cur_id += 1

        from multiprocessing import Pool

        pool = Pool()
        res_single_cells = []
        tmp_df_path = save_dir / Path("tmp_df.csv")
        for res_dict in tqdm.tqdm(pool.imap_unordered(_gen_underseg_scs_sample_wrapper, inputs), total=len(inputs)):
            if not res_dict["is_success"]:
                continue
            _merged_scs = res_dict["merged_scs"]
            cur_merged_sc = res_dict["cur_merged_sc"]
            df = res_dict["df"]
            if all_df is None:
                all_df = df
            else:
                all_df = pd.concat([all_df, df], ignore_index=True)
            counter += 1
            tmp_df_path.parent.mkdir(parents=True, exist_ok=True)
            all_df.to_csv(tmp_df_path, index=False)
        pool.close()
        pool.join()

        # delete tmp_df file
        if tmp_df_path.exists():
            os.remove(tmp_df_path)

        return all_df, counter, cur_id

    if parallel:
        all_df = None
        counter = 0
        cur_id = 0
        while counter < total_sample_num:
            print(
                ">>>>>>> a new round of parallely generating underseg scs - #sample already generated: ",
                counter,
                "cur_id=",
                cur_id,
                "total_sample_num:",
                total_sample_num,
                "<<<<<<<",
            )
            all_df, counter, cur_id = process_parallel(total_sample_num - counter, all_df, counter, cur_id)
        return all_df
    else:
        _process_sequential()
