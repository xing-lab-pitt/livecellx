import itertools
from os import listdir

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.interpolate import bisplev, bisplrep
from skimage import filters, morphology
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.io import imread
from skimage.measure import label, regionprops
from legacy_utils.unsharp_mask import unsharp_mask


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=2, fill_mode="nearest", cval=0.0):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [
        ndimage.interpolation.affine_transform(
            x_channel, final_affine_matrix, final_offset, order=0, mode=fill_mode, cval=cval
        )
        for x_channel in x
    ]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_zoom(x, y, zoom_range, row_axis=0, col_axis=1, channel_axis=2, fill_mode="nearest", cval=0.0):
    """Performs a random spatial zoom of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Zoomed Numpy image tensor.
    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError("zoom_range should be a tuple or list of two floats. " "Received arg: ", zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode="nearest")[:, :, 0]
    y = apply_transform(y, transform_matrix, channel_axis, fill_mode="nearest")[:, :, 0]
    return x, y


def random_flip(x, y):
    flip_axis = np.random.randint(0, high=100) % 2
    x = np.flip(x, flip_axis)
    y = np.flip(y, flip_axis)
    return x, y


# def img_gt_transform(img,gt,clip_low=0.01,clip_high=0.02):

#     img=(img-np.amin(img))*1.0/(np.amax(img)-np.amin(img))#img*1.0 transform array to double
#     eah_flag=np.random.randint(0,high=100)%2
#     if eah_flag==1:
#         clip_value=np.random.uniform(clip_low,high=clip_high)
#         img = equalize_adapthist(img,clip_limit=clip_value)

#     img=img*1.0/np.median(img)
#     img=np.expand_dims(img,axis=2)

#     gt=ndimage.distance_transform_edt(gt)
#     if np.count_nonzero(gt)!=0:
#         nonzero_gt=gt[gt>0]
#         gt=gt*1.0/np.median(nonzero_gt)

#     gt= np.expand_dims(gt,axis=2)

#     return img,gt


def img_transform(img):

    """
    Normalizing the image to its median.
    And expand the image dimension.
    """
    if (np.amax(img) - np.amin(img)) != 0.0 and np.median(img) != 0.0:
        img = (img - np.amin(img)) * 1.0 / (np.amax(img) - np.amin(img))  # img*1.0 transform array to double
        img = img * 1.0 / np.median(img)

    img = img[..., np.newaxis]
    return img


def img_gt_transform(img, gt):

    img = img_transform(img)

    gt = ndimage.distance_transform_edt(gt)
    if np.count_nonzero(gt) != 0:
        nonzero_gt = gt[gt > 0]
        gt = gt * 1.0 / np.median(nonzero_gt)

    gt = np.expand_dims(gt, axis=2)

    return img, gt


# def img_transform(img,img_h,img_w):
#     img=(img-np.amin(img))*1.0/(np.amax(img)-np.amin(img))#img*1.0 transform array to double
#     img=img*1.0/np.median(img)
#     img=np.reshape(img,(img_h,img_w,1))
#     return img


def prep_train_data(img_path, gt_path, edge=0):
    """
    img_path - string, folder containing training images
    gt_path - string, folder containing training answers
    edge - option, if True, using edge info.
    """
    img_list = sorted(listdir(img_path))
    gt_list = sorted(listdir(gt_path))
    imgs, gts = [], []
    for i in range(len(img_list)):
        img = imread(img_path + "/" + img_list[i])
        gt = imread(gt_path + "/" + gt_list[i])
        # print('img shape:', img.shape, 'gt shape:', gt.shape)
        assert img.shape[:2] == gt.shape[:2]
        imgs.append(img)
        gts.append(gt)
    if edge:
        return prep_train_data_list_w_edge(imgs, gts)
    else:
        return prep_train_data_list(imgs, gts)


def prep_train_data_list(imgs, gts):
    assert len(imgs) == len(gts)
    data, label = [], []
    for i in range(len(imgs)):
        # img = imread(img_path +'/'+ img_list[i])
        # gt=imread(gt_path+'/'+gt_list[i])
        img = imgs[i]
        gt = gts[i]

        img_h = img.shape[0]
        img_w = img.shape[1]

        img_blur = gaussian(img)
        img_unsharp = unsharp_mask(img, preserve_range=True)
        img_noise = img + 0.5 * img.std() * np.random.randn(*img.shape)
        img_zoom, gt_zoom = random_zoom(np.expand_dims(img, axis=2), np.expand_dims(gt, axis=2), zoom_range=(0.5, 1))
        img_flip, gt_flip = random_flip(img, gt)

        img_trans, gt_trans = img_gt_transform(img, gt)
        img_blur, gt_blur = img_gt_transform(img_blur, gt)
        img_unsharp, gt_unsharp = img_gt_transform(img_unsharp, gt)
        img_noise, gt_noise = img_gt_transform(img_noise, gt)
        img_zoom, gt_zoom = img_gt_transform(img_zoom, gt_zoom)
        img_flip, gt_flip = img_gt_transform(img_flip, gt_flip)

        data.append(img_trans)
        data.append(img_blur)
        data.append(img_unsharp)
        data.append(img_noise)
        data.append(img_zoom)
        data.append(img_flip)

        label.append(gt_trans)
        label.append(gt_blur)
        label.append(gt_unsharp)
        label.append(gt_noise)
        label.append(gt_zoom)
        label.append(gt_flip)

    data = np.array(data)
    label = np.array(label)
    return data, label


def prep_train_data_list_w_edge(imgs, gts):

    """
    This is for preparing the training date.
    Adding a detected edge part to the original images.
    """
    assert len(imgs) == len(gts)
    data, label = [], []
    for i in range(len(imgs)):
        # img = imread(img_path +'/'+ img_list[i])
        # gt=imread(gt_path+'/'+gt_list[i])
        img = imgs[i]
        gt = gts[i]

        img_blur = gaussian(img)
        img_unsharp = unsharp_mask(img, preserve_range=True)
        img_noise = img + 0.5 * img.std() * np.random.randn(*img.shape)
        img_zoom, gt_zoom = random_zoom(np.expand_dims(img, axis=2), np.expand_dims(gt, axis=2), zoom_range=(0.5, 1))
        img_flip, gt_flip = random_flip(img, gt)

        # adding filter for calculating edges
        img_edge = filters.roberts(img)
        img_blur_edge = filters.roberts(img_blur)
        img_unsharp_edge = filters.roberts(img_unsharp)
        img_noise_edge = filters.roberts(img_noise)
        img_zoom_edge = filters.roberts(img_zoom)
        img_flip_edge = filters.roberts(img_flip)
        # concate with the original images
        img = np.concatenate((img, img_edge), axis=0)
        img_blur = np.concatenate((img_blur, img_blur_edge), axis=0)
        img_unsharp = np.concatenate((img_unsharp, img_unsharp_edge), axis=0)
        img_noise = np.concatenate((img_noise, img_noise_edge), axis=0)
        img_zoom = np.concatenate((img_zoom, img_zoom_edge), axis=0)
        img_flip = np.concatenate((img_flip, img_flip_edge), axis=0)

        gt = np.concatenate((gt, gt), axis=0)
        gt_zoom = np.concatenate((gt_zoom, gt_zoom), axis=0)
        gt_flip = np.concatenate((gt_flip, gt_flip), axis=0)

        # print(img.shape, gt.shape)

        img_trans, gt_trans = img_gt_transform(img, gt)
        img_blur, gt_blur = img_gt_transform(img_blur, gt)
        img_unsharp, gt_unsharp = img_gt_transform(img_unsharp, gt)
        img_noise, gt_noise = img_gt_transform(img_noise, gt)
        img_zoom, gt_zoom = img_gt_transform(img_zoom, gt_zoom)
        img_flip, gt_flip = img_gt_transform(img_flip, gt_flip)

        # print(img_trans.shape, gt_trans.shape)

        data.append(img_trans)
        data.append(img_blur)
        data.append(img_unsharp)
        data.append(img_noise)
        data.append(img_zoom)
        data.append(img_flip)

        label.append(gt_trans)
        label.append(gt_blur)
        label.append(gt_unsharp)
        label.append(gt_noise)
        label.append(gt_zoom)
        label.append(gt_flip)

    data = np.array(data)
    label = np.array(label)
    return data, label


def dic_bg_correction(Img, ordr=1):
    def poly_matrix(x, y, order=1):
        """generate Matrix use with lstsq"""
        ncols = (order + 1) ** 2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order + 1), range(order + 1))
        for k, (i, j) in enumerate(ij):
            G[:, k] = x**i * y**j
        return G

    x, y = np.arange(0, Img.shape[0], 1), np.arange(0, Img.shape[0], 1)
    X, Y = np.meshgrid(x, y)
    # make Matrix:
    G = poly_matrix(X.flatten(), Y.flatten(), ordr)
    # Solve for np.dot(G, m) = z:
    m = np.linalg.lstsq(G, Img.flatten())[0]
    xx, yy = np.meshgrid(x, y)
    GG = poly_matrix(xx.ravel(), yy.ravel(), ordr)
    zz = np.reshape(np.dot(GG, m), xx.shape)

    return zz


def prep_dic_data(img_path, img_list, img_num, bg_corrction_flag=True):
    """
    Function - reformat the image data as the input for the cnn (autoencoder)
    """
    if bg_corrction_flag:
        img0 = np.array(imread(img_path + img_list[0]))
        bg0 = dic_bg_correction(img0, ordr=1)
        bg = bg0 - np.mean(bg0)

    img = imread(img_path + img_list[img_num - 1])
    # print('raw img shape:', img.shape)
    if bg_corrction_flag:
        img = img - bg

    img = (img - np.amin(img)) * 1.0 / (np.amax(img) - np.amin(img))  # img*1.0 transform array to double
    img = img * 1.0 / np.median(img)
    img = np.expand_dims(img, axis=2)
    data = np.expand_dims(img, axis=0)
    return data


def prep_dic_imgs(imgs):
    res = []
    for img in imgs:
        res.append(prep_dic_single_img(img))
    return res


def prep_dic_single_img(img, bg_corrction_flag=True):
    if bg_corrction_flag:
        # img0 = np.array(imread(img_path+img_list[0]))
        bg0 = dic_bg_correction(img, ordr=1)
        bg = bg0 - np.mean(bg0)

    # print('raw img shape:', img.shape)
    if bg_corrction_flag:
        img = img - bg

    img = (img - np.amin(img)) * 1.0 / (np.amax(img) - np.amin(img))  # img*1.0 transform array to double
    img = img * 1.0 / np.median(img)
    img = np.expand_dims(img, axis=2)
    data = np.expand_dims(img, axis=0)
    return data


def fluor_bg_correction(img, r_step=16, c_step=16, smooth_factor=1e10):
    I = img
    n_row, n_col = I.shape[0], I.shape[1]
    ctrl_x = []
    ctrl_y = []
    ctrl_z = []

    for i in np.arange(0, n_row, r_step):
        for j in np.arange(0, n_col, c_step):
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
    tck = bisplrep(ctrl_x, ctrl_y, ctrl_z, s=smooth_factor)
    znew = bisplev(lx, ly, tck)
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     x,y = np.meshgrid(lx,ly)
    #     surf = ax.plot_surface(x, y, znew, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #     plt.show()

    return znew


def prep_fluor_data(img_path, img_list, img_num, bg_corrction_flag=True):

    if bg_corrction_flag:
        img0 = imread(img_path + img_list[0])
        bg0 = fluor_bg_correction(img0)
        bg = bg0 - np.mean(bg0)
    # -----guassian filter is not as good as B-spline-----------
    #     bg0=gaussian(img0,sigma=1000,preserve_range=True)
    #     bg=bg0-np.mean(bg0)

    img = imread(img_path + img_list[img_num - 1])
    if bg_corrction_flag:
        img = img - bg

    img = (img - np.amin(img)) * 1.0 / (np.amax(img) - np.amin(img))  # img*1.0 transform array to double
    img = img * 1.0 / np.median(img)
    img = np.expand_dims(img, axis=2)
    data = np.expand_dims(img, axis=0)
    return data


# def obj_transform(img,obj_h,obj_w,random_eah=True,clip_low=0.01,clip_high=0.03,hist_bins=100):
#     img=(img-np.amin(img))*1.0/(np.amax(img)-np.amin(img))#img*1.0 transform array to double
#     if random_eah==True:
#         eah_flag=np.random.randint(0,high=100)%2
#         if eah_flag==1:
#             clip_value=np.random.uniform(clip_low,high=clip_high)
#             img = equalize_adapthist(img,clip_limit=clip_value,nbins=hist_bins)

#     img=img*1.0/np.median(img)
#     img=cv2.resize(img, (obj_w,obj_h), interpolation=cv2.INTER_CUBIC)# the dimension order in numpy is different with cv2
#     img=np.expand_dims(img,axis=2)
#     return img


def obj_transform(img, random_eah=True, clip_low=0.01, clip_high=0.02, hist_bins=100):
    "The input img is normalized, and change dimention."
    img = (img - np.amin(img)) * 1.0 / (np.amax(img) - np.amin(img))  # img*1.0 transform array to double
    if random_eah:
        "A adaptive histogram equalization."
        eah_flag = np.random.randint(0, high=100) % 2  # produce one random number.
        if eah_flag == 1:
            clip_value = np.random.uniform(clip_low, high=clip_high)
            img = equalize_adapthist(img, clip_limit=clip_value, nbins=hist_bins)

    img = img * 1.0 / np.median(img)
    img = np.expand_dims(img, axis=2)
    return img


def keep_aspect_resize(img, obj_h, obj_w, fill_mode="nearest"):
    # fill_mode:nearest,median,random_low,random_high

    img = img.astype(np.float)
    old_h, old_w = img.shape[0], img.shape[1]

    ratio_h = float(obj_h) / old_h
    ratio_w = float(obj_w) / old_w
    ratio = min([ratio_h, ratio_w])

    new_h = int(old_h * ratio)
    new_w = int(old_w * ratio)

    # the dimension order in numpy is different with cv2
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    delta_h = obj_h - new_h
    delta_w = obj_w - new_w

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    if fill_mode == "nearest":
        inds = ndimage.distance_transform_edt(img == 0, return_distances=False, return_indices=True)
        img = img[tuple(inds)]
    if fill_mode == "median":
        img[img == 0] = np.median(img[img > 0])
    if fill_mode == "random_low":
        lbg_percent = np.random.randint(10, high=40)
        img[img == 0] = np.percentile(img[img > 0], lbg_percent)
    if fill_mode == "random_high":
        hbg_percent = np.random.randint(60, high=90)
        img[img == 0] = np.percentile(img[img > 0], hbg_percent)

    return img


# 1 fill 0 background with different pixel value
# 2 gaussian,unsharp_mask and noise processing
# 2 keep aspect-ratio resize and fill 0 background with nearest pixels
# 3 normailze and equalize_adapthist
def prep_icnn_seg_train_data(img_path, obj_h, obj_w, classes=["single_cell", "cell_fragment", "multi_cells"]):
    data = []
    label = []

    for cla in classes:
        img_sub_path = img_path + "/" + cla
        img_list = sorted(listdir(img_sub_path))
        i = 0
        while i < len(img_list):

            img = imread(img_sub_path + "/" + img_list[i])
            mask = img != 0

            if len(img[mask]) == 0:
                # print(img)
                # print(mask)
                print(img_sub_path + "/" + img_list[i])
                i = i + 1
                continue
            # mask = img > 0

            # fill with nearest neighbor(distance transform)
            img_nbg = np.copy(img)
            img_mbg = np.copy(img)  # fill background with median
            # img_abg=np.copy(img)  # fill with average(mean)
            img_hbg = np.copy(img)  # fill with maximum
            img_lbg = np.copy(img)  # fill with minimum

            inds = ndimage.distance_transform_edt(img_nbg == 0, return_distances=False, return_indices=True)
            img_nbg = img_nbg[tuple(inds)]
            img_blur_nbg = gaussian(img_nbg, sigma=0.5, preserve_range=True)
            img_unsharp_nbg = unsharp_mask(img_nbg, preserve_range=True)
            img_noise_nbg = img_nbg + 0.25 * img_nbg.std() * np.random.randn(*img_nbg.shape)

            img_mbg[img_mbg == 0] = np.median(img[mask])
            img_blur_mbg = gaussian(img_mbg, sigma=0.5, preserve_range=True)
            img_unsharp_mbg = unsharp_mask(img_mbg, preserve_range=True)
            img_noise_mbg = img_mbg + 0.25 * img_mbg.std() * np.random.randn(*img_mbg.shape)

            lbg_percent = np.random.randint(10, high=40)
            img_lbg[img_lbg == 0] = np.percentile(img[mask], lbg_percent)
            # img_blur_lbg=gaussian(img_lbg,sigma=0.5,preserve_range=True)
            img_unsharp_lbg = unsharp_mask(img_lbg, preserve_range=True)
            img_noise_lbg = img_lbg + 0.25 * img_lbg.std() * np.random.randn(*img_lbg.shape)

            hbg_percent = np.random.randint(60, high=90)
            img_hbg[img_hbg == 0] = np.percentile(img[mask], hbg_percent)
            # img_blur_hbg=gaussian(img_hbg,sigma=0.5,preserve_range=True)
            img_unsharp_hbg = unsharp_mask(img_hbg, preserve_range=True)
            img_noise_hbg = img_hbg + 0.25 * img_hbg.std() * np.random.randn(*img_hbg.shape)

            img_nbg = obj_transform(keep_aspect_resize(img_nbg, obj_h, obj_w))
            img_blur_nbg = obj_transform(keep_aspect_resize(img_blur_nbg, obj_h, obj_w))
            img_unsharp_nbg = obj_transform(keep_aspect_resize(img_unsharp_nbg, obj_h, obj_w))
            img_noise_nbg = obj_transform(keep_aspect_resize(img_noise_nbg, obj_h, obj_w))

            img_mbg = obj_transform(keep_aspect_resize(img_mbg, obj_h, obj_w))
            img_blur_mbg = obj_transform(keep_aspect_resize(img_blur_mbg, obj_h, obj_w))
            img_unsharp_mbg = obj_transform(keep_aspect_resize(img_unsharp_mbg, obj_h, obj_w))
            img_noise_mbg = obj_transform(keep_aspect_resize(img_noise_mbg, obj_h, obj_w))

            img_unsharp_lbg = obj_transform(keep_aspect_resize(img_unsharp_lbg, obj_h, obj_w))
            img_noise_lbg = obj_transform(keep_aspect_resize(img_noise_lbg, obj_h, obj_w))

            img_unsharp_hbg = obj_transform(keep_aspect_resize(img_unsharp_hbg, obj_h, obj_w))
            img_noise_hbg = obj_transform(keep_aspect_resize(img_noise_hbg, obj_h, obj_w))

            if cla == classes[0]:
                data.append(img_nbg)
                data.append(img_blur_nbg)
                data.append(img_unsharp_nbg)
                data.append(img_noise_nbg)

                data.append(img_mbg)
                data.append(img_blur_mbg)
                data.append(img_unsharp_mbg)
                data.append(img_noise_mbg)

                data.append(img_unsharp_lbg)
                data.append(img_noise_lbg)

                data.append(img_unsharp_hbg)
                data.append(img_noise_hbg)

                gt = np.array([1, 0, 0])
                for l in range(12):
                    label.append(gt)

            elif cla == classes[1]:
                data.append(img_nbg)
                data.append(img_blur_nbg)
                data.append(img_unsharp_nbg)
                data.append(img_noise_nbg)

                data.append(img_mbg)
                data.append(img_blur_mbg)
                data.append(img_unsharp_mbg)
                data.append(img_noise_mbg)

                data.append(img_unsharp_lbg)
                data.append(img_noise_lbg)

                data.append(img_unsharp_hbg)
                data.append(img_noise_hbg)

                gt = np.array([0, 1, 0])
                for m in range(12):
                    label.append(gt)

            elif cla == classes[2]:
                data.append(img_nbg)
                data.append(img_blur_nbg)
                data.append(img_unsharp_nbg)
                data.append(img_noise_nbg)

                data.append(img_mbg)
                data.append(img_blur_mbg)
                data.append(img_unsharp_mbg)
                data.append(img_noise_mbg)

                data.append(img_unsharp_lbg)
                data.append(img_noise_lbg)

                data.append(img_unsharp_hbg)
                data.append(img_noise_hbg)

                gt = np.array([0, 0, 1])
                for n in range(12):
                    label.append(gt)

            i += 1
    data, label = np.array(data), np.array(label)
    return data, label


def prep_icnn_seg_reduc_aug(img_path, obj_h, obj_w, classes=["single_cell", "cell_fragment", "multi_cells"]):
    data = []
    label = []

    for cla in classes:
        img_sub_path = img_path + "/" + cla
        img_list = sorted(listdir(img_sub_path))
        i = 0
        while i < len(img_list):

            img = imread(img_sub_path + "/" + img_list[i])
            mask = img != 0

            if len(img[mask]) == 0:
                # print(img)
                # print(mask)
                print(img_sub_path + "/" + img_list[i])
                i = i + 1
                continue
            # mask = img > 0

            # fill with nearest neighbor(distance transform)
            img_nbg = np.copy(img)
            img_mbg = np.copy(img)  # fill background with median
            # img_abg=np.copy(img)  # fill with average(mean)
            img_hbg = np.copy(img)  # fill with maximum
            img_lbg = np.copy(img)  # fill with minimum

            inds = ndimage.distance_transform_edt(img_nbg == 0, return_distances=False, return_indices=True)
            img_nbg = img_nbg[tuple(inds)]
            img_blur_nbg = gaussian(img_nbg, sigma=0.5, preserve_range=True)
            img_unsharp_nbg = unsharp_mask(img_nbg, preserve_range=True)
            img_noise_nbg = img_nbg + 0.25 * img_nbg.std() * np.random.randn(*img_nbg.shape)

            img_mbg[img_mbg == 0] = np.median(img[mask])
            img_blur_mbg = gaussian(img_mbg, sigma=0.5, preserve_range=True)
            img_unsharp_mbg = unsharp_mask(img_mbg, preserve_range=True)
            img_noise_mbg = img_mbg + 0.25 * img_mbg.std() * np.random.randn(*img_mbg.shape)

            lbg_percent = np.random.randint(10, high=40)
            img_lbg[img_lbg == 0] = np.percentile(img[mask], lbg_percent)
            # img_blur_lbg=gaussian(img_lbg,sigma=0.5,preserve_range=True)
            img_unsharp_lbg = unsharp_mask(img_lbg, preserve_range=True)
            img_noise_lbg = img_lbg + 0.25 * img_lbg.std() * np.random.randn(*img_lbg.shape)

            hbg_percent = np.random.randint(60, high=90)
            img_hbg[img_hbg == 0] = np.percentile(img[mask], hbg_percent)
            # img_blur_hbg=gaussian(img_hbg,sigma=0.5,preserve_range=True)
            img_unsharp_hbg = unsharp_mask(img_hbg, preserve_range=True)
            img_noise_hbg = img_hbg + 0.25 * img_hbg.std() * np.random.randn(*img_hbg.shape)

            img_nbg = obj_transform(keep_aspect_resize(img_nbg, obj_h, obj_w))
            img_blur_nbg = obj_transform(keep_aspect_resize(img_blur_nbg, obj_h, obj_w))
            img_unsharp_nbg = obj_transform(keep_aspect_resize(img_unsharp_nbg, obj_h, obj_w))
            img_noise_nbg = obj_transform(keep_aspect_resize(img_noise_nbg, obj_h, obj_w))

            img_mbg = obj_transform(keep_aspect_resize(img_mbg, obj_h, obj_w))
            img_blur_mbg = obj_transform(keep_aspect_resize(img_blur_mbg, obj_h, obj_w))
            img_unsharp_mbg = obj_transform(keep_aspect_resize(img_unsharp_mbg, obj_h, obj_w))
            img_noise_mbg = obj_transform(keep_aspect_resize(img_noise_mbg, obj_h, obj_w))

            img_unsharp_lbg = obj_transform(keep_aspect_resize(img_unsharp_lbg, obj_h, obj_w))
            img_noise_lbg = obj_transform(keep_aspect_resize(img_noise_lbg, obj_h, obj_w))

            img_unsharp_hbg = obj_transform(keep_aspect_resize(img_unsharp_hbg, obj_h, obj_w))
            img_noise_hbg = obj_transform(keep_aspect_resize(img_noise_hbg, obj_h, obj_w))

            if cla == classes[0]:
                data.append(img_nbg)
                data.append(img_blur_nbg)
                data.append(img_unsharp_nbg)
                data.append(img_noise_nbg)

                data.append(img_mbg)
                data.append(img_blur_mbg)
                data.append(img_unsharp_mbg)
                data.append(img_noise_mbg)

                data.append(img_unsharp_lbg)
                data.append(img_noise_lbg)

                data.append(img_unsharp_hbg)
                data.append(img_noise_hbg)

                gt = np.array([1, 0, 0])
                for l in range(12):
                    label.append(gt)

            elif cla == classes[1]:
                data.append(img_nbg)
                data.append(img_blur_nbg)
                data.append(img_unsharp_nbg)
                data.append(img_noise_nbg)

                data.append(img_mbg)
                data.append(img_blur_mbg)
                data.append(img_unsharp_mbg)
                data.append(img_noise_mbg)

                data.append(img_unsharp_lbg)
                data.append(img_noise_lbg)

                data.append(img_unsharp_hbg)
                data.append(img_noise_hbg)

                gt = np.array([0, 1, 0])
                for m in range(12):
                    label.append(gt)

            elif cla == classes[2]:
                data.append(img_nbg)
                data.append(img_blur_nbg)
                data.append(img_unsharp_nbg)
                data.append(img_noise_nbg)

                data.append(img_mbg)
                data.append(img_blur_mbg)
                data.append(img_unsharp_mbg)
                data.append(img_noise_mbg)

                data.append(img_unsharp_lbg)
                data.append(img_noise_lbg)

                data.append(img_unsharp_hbg)
                data.append(img_noise_hbg)

                gt = np.array([0, 0, 1])
                for n in range(12):
                    label.append(gt)

            i += 1
    data, label = np.array(data), np.array(label)
    return data, label


# 1 keep aspect-ratio resize and fill 0 background with different pixel value
# 2 gaussian,unsharp_mask and noise processing
# 3 normailze and equalize_adapthist


def prep_icnn_am_train_data(img_path, obj_h, obj_w):
    """
    img_path: a folder contining the /other, /apoptosis, /mitosis folders.
    obj_h and obj_w is defined dimention of input pictures.
    """
    data = []
    label = []
    classes = ["other", "apoptosis", "mitosis"]
    for cla in classes:
        sub_path = img_path + "/" + cla
        img_list = sorted(listdir(sub_path))
        i = 0
        while i < len(img_list):

            img = imread(sub_path + "/" + img_list[i])
            img_nbg = keep_aspect_resize(img, obj_h, obj_w, fill_mode="nearest")
            img_mbg = keep_aspect_resize(img, obj_h, obj_w, fill_mode="median")
            img_lbg = keep_aspect_resize(img, obj_h, obj_w, fill_mode="random_low")
            img_hbg = keep_aspect_resize(img, obj_h, obj_w, fill_mode="random_high")

            img_blur_nbg = obj_transform(gaussian(img_nbg, sigma=0.5, preserve_range=True))
            img_blur_mbg = obj_transform(gaussian(img_mbg, sigma=0.5, preserve_range=True))
            img_blur_lbg = obj_transform(gaussian(img_lbg, sigma=0.5, preserve_range=True))
            img_blur_hbg = obj_transform(gaussian(img_hbg, sigma=0.5, preserve_range=True))

            img_unsharp_nbg = obj_transform(unsharp_mask(img_nbg, preserve_range=True))
            img_unsharp_mbg = obj_transform(unsharp_mask(img_mbg, preserve_range=True))
            img_unsharp_lbg = obj_transform(unsharp_mask(img_lbg, preserve_range=True))
            img_unsharp_hbg = obj_transform(unsharp_mask(img_hbg, preserve_range=True))

            img_noise_nbg = obj_transform(img_nbg + 0.25 * img_nbg.std() * np.random.randn(*img_nbg.shape))
            img_noise_mbg = obj_transform(img_mbg + 0.25 * img_mbg.std() * np.random.randn(*img_mbg.shape))
            img_noise_lbg = obj_transform(img_lbg + 0.25 * img_lbg.std() * np.random.randn(*img_lbg.shape))
            img_noise_hbg = obj_transform(img_hbg + 0.25 * img_hbg.std() * np.random.randn(*img_hbg.shape))

            img_nbg = obj_transform(img_nbg)
            img_mbg = obj_transform(img_mbg)
            img_lbg = obj_transform(img_lbg)
            img_hbg = obj_transform(img_hbg)

            if cla == "other":
                data.append(img_nbg)
                data.append(img_blur_nbg)
                data.append(img_unsharp_nbg)
                data.append(img_noise_nbg)

                data.append(img_mbg)
                data.append(img_blur_mbg)
                data.append(img_unsharp_mbg)
                data.append(img_noise_mbg)

                # data.append(img_hbg)
                # data.append(img_blur_hbg)
                data.append(img_unsharp_hbg)
                data.append(img_noise_hbg)

                # data.append(img_lbg)
                # data.append(img_blur_lbg)
                data.append(img_unsharp_lbg)
                data.append(img_noise_lbg)

                gt = np.array([1, 0, 0])
                for l in range(12):
                    label.append(gt)

            elif cla == "apoptosis":
                data.append(img_nbg)
                data.append(img_blur_nbg)
                data.append(img_unsharp_nbg)
                data.append(img_noise_nbg)

                data.append(img_mbg)
                data.append(img_blur_mbg)
                data.append(img_unsharp_mbg)
                data.append(img_noise_mbg)

                # data.append(img_hbg)
                # data.append(img_blur_hbg)
                data.append(img_unsharp_hbg)
                data.append(img_noise_hbg)

                # data.append(img_lbg)
                # data.append(img_blur_lbg)
                data.append(img_unsharp_lbg)
                data.append(img_noise_lbg)

                gt = np.array([0, 1, 0])
                for m in range(12):
                    label.append(gt)
            elif cla == "mitosis":
                data.append(img_nbg)
                data.append(img_blur_nbg)
                data.append(img_unsharp_nbg)
                data.append(img_noise_nbg)

                data.append(img_mbg)
                data.append(img_blur_mbg)
                data.append(img_unsharp_mbg)
                data.append(img_noise_mbg)

                # data.append(img_hbg)
                # data.append(img_blur_hbg)
                data.append(img_unsharp_hbg)
                data.append(img_noise_hbg)

                # data.append(img_lbg)
                # data.append(img_blur_lbg)
                data.append(img_unsharp_lbg)
                data.append(img_noise_lbg)

                gt = np.array([0, 0, 1])
                for n in range(12):
                    label.append(gt)

            i += 1
    data, label = np.array(data), np.array(label)
    return data, label


# gererate single segmented img with its surrounding with filled with the
# nearest pixel values


def generate_single_cell_img_edt(img_path, seg_path, img_list, seg_img_list, obj_h, obj_w, img_num, obj_num):
    img = imread(img_path + "/" + img_list[img_num - 1])
    seg_img = imread(seg_path + "/" + seg_img_list[img_num - 1])

    # single_obj_img=morphology.binary_dilation(seg_img==obj_num,morphology.diamond(16))
    single_obj_img = seg_img == obj_num
    single_obj_img = label(single_obj_img)
    rps = regionprops(single_obj_img)
    candi_r = [r for r in rps if r.label == 1][0]
    candi_box = candi_r.bbox
    single_cell_img = single_obj_img * img
    crop_img = single_cell_img[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]

    inds = ndimage.distance_transform_edt(crop_img == 0, return_distances=False, return_indices=True)
    crop_img = crop_img[tuple(inds)]
    crop_img = obj_transform(keep_aspect_resize(crop_img, obj_h, obj_w), random_eah=False)
    crop_img = np.expand_dims(crop_img, axis=0)
    return crop_img


# gererate single segmented img with its surrounding with filled with the
# nearest pixel values for testing


def generate_single_cell_img_edt_test(img_path, seg_path, img_list, seg_img_list, obj_h, obj_w, img_num, obj_num):
    img = imread(img_path + "/" + img_list[img_num - 1])
    seg_img = imread(seg_path + "/" + seg_img_list[img_num - 1])

    # single_obj_img=morphology.binary_dilation(seg_img==obj_num,morphology.diamond(16))
    single_obj_img = seg_img == obj_num
    single_obj_img = label(single_obj_img)
    rps = regionprops(single_obj_img)
    candi_r = [r for r in rps if r.label == 1][0]
    candi_box = candi_r.bbox
    single_cell_img = single_obj_img * img
    crop_img = single_cell_img[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]

    inds = ndimage.distance_transform_edt(crop_img == 0, return_distances=False, return_indices=True)
    crop_img = crop_img[tuple(inds)]

    plt.imshow(crop_img)
    plt.show()

    crop_img = obj_transform(keep_aspect_resize(crop_img, obj_h, obj_w), random_eah=False)
    crop_img = np.expand_dims(crop_img, axis=0)
    return crop_img


# crop single segmented img by bounding box from the original image


def generate_single_cell_img_env(img, rps, obj_h, obj_w, obj_num):

    candi_r = [r for r in rps if r.label == obj_num][0]
    candi_box = candi_r.bbox
    crop_img = img[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]

    crop_img = obj_transform(keep_aspect_resize(crop_img, obj_h, obj_w), random_eah=False)
    crop_img = np.expand_dims(crop_img, axis=0)
    return crop_img


# crop single segmented img by bounding box from the original image for test


def generate_single_cell_img_env_test(img, seg_img, obj_h, obj_w, obj_num):
    # single_obj_img=morphology.binary_dilation(seg_img==obj_num,morphology.diamond(16))
    single_obj_img = seg_img == obj_num
    single_obj_img = label(single_obj_img)
    rps = regionprops(single_obj_img)
    candi_r = [r for r in rps if r.label == 1][0]
    candi_box = candi_r.bbox
    crop_img = img[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]

    plt.imshow(crop_img)
    plt.show()

    crop_img = obj_transform(keep_aspect_resize(crop_img, obj_h, obj_w), random_eah=False)
    crop_img = np.expand_dims(crop_img, axis=0)

    return crop_img

    # plt.imshow(img_unsharp[:,:,0])
    # plt.show()

    # plt.imshow(gt_unsharp[:,:,0])
    # plt.show()

    # plt.imshow(img_zoom[:,:,0])
    # plt.show()

    # plt.imshow(gt_zoom[:,:,0])
    # plt.show()

    # plt.imshow(img_flip[:,:,0])
    # plt.show()

    # plt.imshow(gt_flip[:,:,0])
    # plt.show()

    # plt.imshow(img)
    # plt.show()

    # plt.title('nearest')
    # plt.imshow(img_nbg[:,:,0])
    # plt.show()

    # plt.imshow(img_blur_nbg[:,:,0])
    # plt.show()

    # plt.imshow(img_unsharp_nbg[:,:,0])
    # plt.show()

    # plt.imshow(img_noise_nbg[:,:,0])
    # plt.show()

    # print(img_mbg.shape)
    # plt.title('median')
    # plt.imshow(img_mbg[:,:,0])
    # plt.show()

    # plt.imshow(img_blur_mbg[:,:,0])
    # plt.show()

    # plt.imshow(img_unsharp_mbg[:,:,0])
    # plt.show()

    # plt.imshow(img_noise_mbg[:,:,0])
    # plt.show()

    # plt.title('low')
    # plt.imshow(img_lbg[:,:,0])
    # plt.show()

    # plt.imshow(img_blur_lbg[:,:,0])
    # plt.show()

    # plt.imshow(img_unsharp_lbg[:,:,0])
    # plt.show()

    # plt.imshow(img_noise_lbg[:,:,0])
    # plt.show()

    # plt.title('high')
    # plt.imshow(img_hbg[:,:,0])
    # plt.show()

    # plt.imshow(img_blur_hbg[:,:,0])
    # plt.show()

    # plt.imshow(img_unsharp_hbg[:,:,0])
    # plt.show()

    # plt.imshow(img_noise_hbg[:,:,0])
    # plt.show()
