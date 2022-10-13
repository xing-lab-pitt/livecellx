import math
import os
import subprocess
import uuid

import cv2
import cv2 as cv
import legacy_utils.cv_configs as cv_configs
import matplotlib.cm
import numpy as np
import PIL
import scipy
import skimage
import skimage.segmentation
import sklearn
import legacy_utils.utils as utils
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import feature, measure
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.filters import threshold_local
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from legacy_utils.utils import distance


def denoise_images(images, win_size=21):
    res = []
    o_shape = images.shape
    images = images.reshape([-1] + list(o_shape[-2:]))
    for image in images:
        # image = cv2.fastNlMeansDenoising(image, None, 20, 7, 21)
        # image = utils.denoise_wavelet(image)
        # image = utils.denoise_nl_means(image)
        image = utils.denoise_bilateral(image, win_size=win_size)
        res.append(image)
    return np.array(res).reshape(o_shape)


def find_image_edges(images):
    res = []
    for image in images:
        not_sure_threshold = 2
        sure_threshold = 6
        # image = cv2.fastNlMeansDenoising(image,None, 20, 7, 21)
        # edges = cv2.Canny(image, not_sure_threshold, sure_threshold, 20)
        edges = feature.canny(image, low_threshold=not_sure_threshold, high_threshold=sure_threshold)
        edges[edges == 0] = np.ma.masked
        res.append(edges)
    return res


def collapse_centers(centers, groups, max_group_sample_size=None, dist_threshold=60, image=None):
    """
    collapse centers based on sorted (prior: high to low) list of center
    use brightest pixel as center group representative
    """
    cur_centers = list(centers)
    cur_groups = [list(group) for group in groups]

    while True:
        temp_centers = []
        temp_groups = []
        marked = [False for _ in range(len(cur_centers))]
        any_collapsed = False
        for i in range(0, len(cur_centers)):
            if marked[i]:
                continue
            center = cur_centers[i]
            group = cur_groups[i]
            collapsed = False
            for j in range(i + 1, len(cur_centers)):
                if marked[j]:
                    continue
                other = cur_centers[j]
                other_group = cur_groups[j]
                if distance(center, other) < dist_threshold:
                    temp_groups.append(group + other_group)
                    # based on biased group. prioritize brightest centers
                    # new_center = np.mean(temp_groups[-1][:max_group_sample_size],
                    #                      axis=0)
                    new_center = temp_groups[-1][0]  # use brightest point
                    temp_centers.append(new_center)
                    collapsed = True
                    any_collapsed = True
                    marked[j] = True
                    break

            if not collapsed:
                temp_centers.append(center)
                temp_groups.append(group)

        if not any_collapsed:
            break
        else:
            cur_centers = temp_centers
            cur_groups = temp_groups

    return cur_centers, cur_groups


def cluster_indices(pts, image, center_num=10, dist_threshold=60, max_group_sample_size=1000):
    # print('#selected pixels:', len(pts))
    origins = pts
    pts = list(pts)
    min_step = 200
    max_step = 100000
    i = 0
    z_tol = 1
    marked = [False for _ in range(len(pts))]
    marked_id = set()
    centers = []
    groups = []
    while i < len(pts) and center_num > len(centers):
        window = []
        window_vals = []
        while len(window) < min_step and i < len(pts):
            if not marked[i]:
                window.append(pts[i])
                window_vals.append(image[pts[i][0], pts[i][1]])
            i += 1
        std = np.std(window_vals, axis=0)
        mean = np.mean(window_vals, axis=0)

        while len(window) < max_step and i < len(pts):
            index = pts[i]
            val = image[index[0], index[1]]
            if std != 0 and (val - mean) / std < z_tol:
                i += 1
                window.append(index)
            else:
                break

        new_centers, new_groups = k_cluster_sklearn(window, k=center_num)

        # NOTE: order of + is important here: keep sorted
        centers = centers + new_centers
        groups = groups + new_groups
        centers, groups = collapse_centers(centers, groups, max_group_sample_size, dist_threshold=dist_threshold)

    return centers[:center_num], groups[:center_num]


def find_chrom_centers_simple_bound(
    image, bounding_size, max_center_num=10, sampling_quantile=0.97, collapse_dist_threshold=20
):
    shape = image.shape
    # print('calculating scores with bounding size=%d...' % bounding_size)
    # scores = find_bounding_sums_cpp(image, bounding_size)
    # scores = image
    scores = utils.denoise_bilateral(image)
    # utils.print_np_vec(scores)
    args = np.argsort(-scores, None)  # sort as flattened
    # C language (row major order)
    # choose top sampling_quantile coordinates regarding intensity,
    # then cluster indices based on these coordinates
    indices = (np.array(np.unravel_index(args, shape, order="C"))).T
    sample_size = utils.count_image_gaussian_quantile(image, sampling_quantile)
    centers, groups = cluster_indices(
        indices[:sample_size], image, center_num=max_center_num, dist_threshold=collapse_dist_threshold
    )
    return centers[:max_center_num], groups[:max_center_num]


def crop_by_sampling(image, sampling_quantile=0.99):
    # scores = utils.denoise_bilateral(image)
    scores = -image
    args = np.argsort(scores, None)
    # indices = (np.array(np.unravel_index(args, image.shape, order='C'))).T
    sample_size = utils.count_image_gaussian_quantile(image, q=sampling_quantile)
    # rows, cols = [x[0] for x in indices], [x[1] for x in indices]
    # rows, cols = rows[:sample_size], cols[:sample_size]
    mask = np.zeros(image.shape[0] * image.shape[1])
    mask[args[:sample_size]] = 1
    mask = mask.reshape(image.shape)
    return mask


def crop_by_local_threshold(image, block_size=301, offset=10):
    # image = utils.denoise_bilateral(image)
    threshold = threshold_local(image, block_size=block_size, offset=offset)
    mask = (image > threshold).astype(int)
    return mask


def bg_correction(image):
    # adapt code from weikang
    sample_step = 5
    I = image
    n_row, n_col = I.shape
    ctrl_x = []
    ctrl_y = []
    ctrl_z = []

    for i in np.arange(0, n_row, sample_step):
        for j in np.arange(0, n_col, sample_step):
            ctrl_x.append(i)
            ctrl_y.append(j)
            ctrl_z.append(I[i, j])
    ctrl_x = np.array(ctrl_x)
    ctrl_y = np.array(ctrl_y)
    ctrl_z = np.array(ctrl_z)

    nx, ny = I.shape[0], I.shape[1]
    lx = np.linspace(0, n_row, nx)
    ly = np.linspace(0, n_col, ny)

    # s value is important for smoothing
    tck = scipy.interpolate.bisplrep(ctrl_x, ctrl_y, ctrl_z, s=1e20)
    znew = scipy.interpolate.bisplev(lx, ly, tck)
    # func = scipy.interpolate.interp2d(ctrl_x, ctrl_y, ctrl_z, kind='quintic')
    # znew = func(lx, ly).T
    res = I - znew
    res[res < 0] = 0
    # plt.imshow(I - znew * 2)
    # fig, axes = plt.subplots(2,2)
    # axes[0, 0].imshow(I)
    # axes[0, 1].imshow(znew)
    # axes[1, 0].imshow(I-znew)
    # axes[1, 1].imshow(I-znew * 2)

    # axes[0, 0].set_title('original')
    # axes[0, 1].set_title('fitted surface')
    # axes[1, 0].set_title('image-surface')
    # axes[1, 1].set_title('image-surfaceXfactor')

    # plt.show()

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # # ax = fig.gca(projection='3d')

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # x, y = np.meshgrid(lx, ly)
    # surf = ax.plot_surface(x, y, znew, cmap=matplotlib.cm.coolwarm,
    #                        linewidth=0)
    # ax.scatter(x, y, I - znew)
    # plt.show()
    return res


def bg_correction_whole_image(image, centers, stats):
    bds = set()
    image = np.copy(image)
    for stat in stats:
        xy_bds = tuple(stat[cv_configs.X_BOX_IND : cv_configs.X_BOX_IND + 4])
        xy_bds = tuple([int(x) for x in xy_bds])
        x1, x2, y1, y2 = xy_bds
        bds.add(xy_bds)

    for xy_bds in bds:
        x1, x2, y1, y2 = xy_bds
        try:
            image[x1:x2, y1:y2] = bg_correction(image[x1:x2, y1:y2])
        except Exception as e:
            print("bisplrep raises err, disregard...")
            print(e)
    return image


def find_contours(image, level=0.5):
    contours = measure.find_contours(image, level)
    return contours


def k_cluster_sklearn(pts, k=10, iters=3, min_group_size=None):
    """
    Note: pts are sorted based on image pixel values.
    return: a list of centers and a list of groups (list of list)
    """
    if len(pts) < k:
        # samples not enough
        return [], []
    # model = sklearn.cluster.KMeans(n_clusters=k)
    model = sklearn.mixture.GaussianMixture(n_components=k)
    cluster_ids = model.fit_predict(pts)
    groups = {}

    # note that group are in order of max -> min intensity
    for i in range(len(cluster_ids)):
        index = cluster_ids[i]
        if not (index in groups):
            groups[index] = []
        groups[index].append(pts[i])

    centers = []
    group_list = []
    for index in groups:
        group = groups[index]
        if min_group_size and len(group) < min_group_size:
            continue
        # center = np.mean(group, axis=0)
        center = group[0]  # use the highest val pixel as representative center
        centers.append(center)
        group_list.append(group)

    return centers, group_list


def find_3d_centers_in_frame_simple_z(image, data, dist_threshold):
    """
    image: Z x X x Y
    """
    centers = []
    center2stats = {}
    for z, c, z_centers, stats in data:
        d3_centers = [center + [z] for center in z_centers]
        d3_centers = [[int(x) for x in center] for center in d3_centers]
        centers += d3_centers
        for i in range(len(d3_centers)):
            center = d3_centers[i]
            center2stats[tuple(center)] = stats[i]
    centers = sorted(centers, key=lambda x: image[x[2], x[0], x[1]], reverse=True)
    # print('#centers in all z axis:', len(centers))
    groups = [[center] for center in centers]
    res_centers, groups = collapse_centers(centers, groups, dist_threshold=dist_threshold)
    max_centers = []
    avg_centers = []

    for i in range(len(groups)):
        group = groups[i]
        pixels = [image[c[2], c[0], c[1]] for c in group]
        # print('debug find 3D pts:', pixels, np.argmax(pixels))
        max_center = group[np.argmax(pixels)]
        # print('max center:', max_center, 'returned by collapse center:', res_centers[i])
        max_center = [int(x) for x in max_center]
        max_centers.append(max_center)
        assert int(distance(max_center, res_centers[i])) == 0
        avg_center = np.mean(group, axis=0)
        avg_centers.append(avg_center)

    # print('%d avg centers, %d max_centers' % (len(avg_centers), len(max_centers)))
    # stats = [center2stats[tuple(center)] for center in max_centers]
    return [max_centers, center2stats]


def get_cropped_images(images, bd):
    """
    Use numpy broadcast to get crop images along last dim axis.
    :param images: 3d or 2d images
    :param bd: a list of boundary min and max [min0, max0, min1, max1, ....]
    :return: a list of cropped images [image0, image1, image2, ...]
    """
    ind_tuple = []
    dims = int(np.array(bd).shape[0] / 2)
    for dim in range(dims):
        ind_tuple.append(slice(bd[dim * 2], bd[dim * 2 + 1] + 1, 1))
    # print('crop slices:', ind_tuple)
    return images[tuple(ind_tuple)]


def get_3d_mask_felzenszwalb(images):
    """
    3d version of felzenzwalb simply by apply felzenswalb to each
    2d image and make label in each 2d image unique.
    not a true "3d fel" segmentation.
    :param images:
    :return:
    """
    mask = []
    label_num = 0
    for z in range(images.shape[2]):
        print("felzenszwalb handling %d of %d images" % (z, images.shape[2]))
        image = images[:, :, z]
        z_mask = skimage.segmentation.felzenszwalb(image, scale=20)
        z_mask += label_num
        label_num += len(set(z_mask.flatten()))
        mask.append(z_mask)
    mask = np.moveaxis(np.array(mask), 0, -1)
    return mask


def get_all_cropped_images(images, bds):
    res = []
    for bd in bds:
        # print('bd:', bd)
        cropped_images = np.array(get_cropped_images(images, bd))
        res.append(cropped_images)
    return res


def max_project_images_2d(images):
    """
    input: images: t x z x c x X x y
    return: t x c x X x Y
    """
    ts, zs, cs, xs, ys = images.shape
    projection = np.zeros(
        (
            ts,
            cs,
            xs,
            ys,
        )
    )
    for t in range(ts):
        for c in range(cs):
            images_3d = images[t, :, c, :, :]
            projection[t, c, :, :] = np.max(images_3d, axis=0)
    return projection


def max_project_images_xyz(images):
    """
    :param images:X x Y x Z
    :return: X x Y projection
    """
    projection = np.max(images, axis=2)
    return projection


def filter_regions_by_area(regions, min_area, max_area):
    res = []
    for region in regions:
        area = region.filled_area
        if area < min_area or area > max_area:
            continue
        res.append(region)
    return res


def find_cells_directly_in_prob_map(prob_map):
    if len(prob_map.shape) == 3:
        prob_map = prob_map[..., 2]
    if len(prob_map.shape) != 2:
        assert False

    prob_map[prob_map < 0.5] = 0
    cell_seg = prob_map
    cell_seg_labels = skimage.segmentation.felzenszwalb(cell_seg, scale=20000)
    # cell_seg_labels = skimage.segmentation.slic(cell_seg.astype(np.double),
    #                                             n_segments=700)
    regions = measure.regionprops(cell_seg_labels, intensity_image=cell_seg)
    regions = filter_regions_by_area(regions, cv_configs.min_cell_area, cv_configs.max_cell_area)
    return cell_seg_labels, regions


def find_signal_directly_in_prob_map(prob_map):
    if len(prob_map.shape) == 3:
        prob_map = prob_map[..., 0]
    if len(prob_map.shape) != 2:
        assert False
    prob_map[prob_map < 0.5] = 0
    coords = skimage.feature.peak_local_max(prob_map, min_distance=cv_configs.collapse_dist_threshold, indices=True)
    coords = sorted(coords, key=lambda x: prob_map[x[0], x[1]], reverse=True)
    groups = [[center] for center in coords]
    coords, groups = collapse_centers(coords, groups, dist_threshold=cv_configs.collapse_dist_threshold)
    return coords


def find_signal_directly_in_tzc_prob_map(tzc_map):
    res = {}
    for t, z, c in tzc_map:
        res[t, z, c] = find_signal_directly_in_prob_map(tzc_map[t, z, c])
    return res


def find_cells_directly_in_tzc_prob_map(tzc_map):
    res_regions = {}
    res_label_masks = {}
    for t, z, c in tzc_map:
        res_label_masks[t, z, c], res_regions[t, z, c] = find_cells_directly_in_prob_map(tzc_map[t, z, c][..., 2])

    return res_label_masks, res_regions


def segment_image_to_small_pieces(image, sh, sw, return_offset=False, start_h_offset=0, start_w_offset=0):
    """

    :param image:
    :param h: smaller h expected
    :param w:
    :return: offset: offset in original image space
    """
    h, w = image.shape[:2]
    res = []
    offsets = []
    # for i in range(start_h_offset, h - sh + 1, sh):
    #     for j in range(start_w_offset, w - sw + 1, sw):
    for i in range(start_h_offset, h, sh):
        for j in range(start_w_offset, w, sw):

            if i + sh <= h and j + sw <= w:
                res.append(image[i : i + sh, j : j + sw, ...])
                offsets.append([i, j])
            else:
                h_lb, h_ub, w_lb, w_ub = i, i + sh, j, j + sw
                if i + sh > h:
                    h_lb = h - sh
                    h_ub = h
                if j + sw > w:
                    w_lb = w - sw
                    w_ub = w
                res.append(image[h_lb:h_ub, w_lb:w_ub, ...])
                offsets.append([h_lb, w_lb])
    if return_offset:
        return res, offsets
    return res


def segment_images_to_small_pieces(images, sh, sw, return_offset=False):
    res = []
    offsets = []
    images = np.array(images)
    for i in range(len(images)):
        # print(images.shape)
        if not return_offset:
            smaller_images = segment_image_to_small_pieces(images[i, ...], sh, sw, return_offset=return_offset)
            res.extend(smaller_images)
        else:
            smaller_images, smaller_offsets = segment_image_to_small_pieces(
                images[i, ...], sh, sw, return_offset=return_offset
            )
            res.extend(smaller_images)
            offsets.append(smaller_offsets)

    if return_offset:
        return np.array(res), np.array(offsets)
    return np.array(res)


def map_single_frame_signal_coord_to_cells(signal_coord, cur_t, cell_mappings):
    """
    returns: cell id, time at which this cell (nearest)
    """
    for t in range(cur_t, -1, -1):
        cell_map = cell_mappings[t]
        for cell_id in cell_map:
            cell = cell_map[cell_id]
            if cell.is_in_cell(signal_coord):
                return cell
    return None


def map_signal_ids_to_cells(signal_mappings, cell_mappings):
    assert len(signal_mappings) == len(cell_mappings)
    T = len(signal_mappings)
    t_trackId2cell = []
    for t in range(T):
        signal_map = signal_mappings[t]
        t_trackId2cell.append({})
        for signal_id in signal_map:
            coord = signal_map[signal_id]
            cell = map_single_frame_signal_coord_to_cells(coord, t, cell_mappings)
            t_trackId2cell[t][signal_id] = cell
    return t_trackId2cell


def normalize_images(Y):
    """
    :param Y: images, N x w x h x n_class
    :return: a normalized version of Y
    """
    Y = np.array(Y)
    res = []
    channel_num = Y.shape[-1]
    for i in range(len(Y)):
        y = Y[i]
        # normalize channels separately
        y_re = y.reshape((-1, y.shape[-1]), order="F")
        mean = np.mean(y_re, axis=0)
        std = np.std(y_re, axis=0)
        y = (y - mean) / std
        res.append(y)
    return np.array(res)


def resize_image_to_2_powers(image):
    closest_size = 2 ** int(math.ceil(math.log2(image.shape[0])))
    # closest_size = 2 ** int(math.floor(math.log2(image.shape[0])))
    resized_img = skimage.transform.resize(image, (closest_size, closest_size))
    return resized_img


def get_CNN_regression_mask(image, relu_edt=False, model=None):

    if model is None:
        model = cv_configs.model

    normalized_image = normalize_images([image])[0]
    resized_img = resize_image_to_2_powers(normalized_image)
    # TODO fix model below
    predicted_mask = model.predict(np.array([resized_img[..., np.newaxis]]), batch_size=1)[0]
    print("predicted map shape:", predicted_mask.shape)
    predicted_mask = predicted_mask.reshape(predicted_mask.shape[:-1])
    predicted_mask = skimage.transform.resize(predicted_mask, image.shape)
    # try old felzenswalb
    # mask = skimage.segmentation.felzenszwalb(predicted_mask, scale=20, min_size=utils.min_cell_area)
    # watershed segmentation
    edt_dist = None
    if not relu_edt:
        possible_cell_mask = predicted_mask > 0.5  # threshold
        edt_dist = ndi.distance_transform_edt(possible_cell_mask)
    else:
        possible_cell_mask = predicted_mask
        edt_dist = predicted_mask
    markers = skimage.feature.peak_local_max(edt_dist, min_distance=20, threshold_abs=3, indices=False)
    markers = ndi.label(markers)[0]
    watershed_mask = skimage.segmentation.watershed(-edt_dist, markers=markers, mask=possible_cell_mask)

    # simple visualization check
    # utils.show_images([image, normalized_image, resized_img, predicted_mask, mask, possible_cell_mask, watershed_mask])
    return watershed_mask


def correct_tiling_gap(image, row_num=3, col_num=3):
    height, width = image.shape[:2]
    tile_h, tile_w = height // row_num, width // col_num
    bg_corrected_img = bg_correction(image)
    return bg_corrected_img


if __name__ == "__main__":
    # BGR order
    # boundary = [(0, 0, 100), (50, 50, 255)]
    # mask_edge_example(img_path, boundary`)
    pass
