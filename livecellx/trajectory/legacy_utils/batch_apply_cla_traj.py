#!/usr/bin/env python
# coding: utf-8
# TODO refactor this module

# In[4]:


import glob
import itertools
import os
import sys
from os import listdir

import cv2
import numpy as np
import pandas as pd
from .cla_seg_model import cla_seg
from keras import optimizers
from keras.layers import Activation, BatchNormalization, Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D
from keras.models import Model
from PIL import Image
from .reg_seg_model import reg_seg
from scipy import signal
from scipy.ndimage import distance_transform_edt, filters
from skimage.io import imread, imread_collection

# In[5]:


main_path = "/home/zoro/Desktop/experiment_data/2018-08-18_HK2_nc_ratio"
input_path = main_path + "/img/"
output_path = main_path + "/output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

posi_end = 20


weight_file = "HK2_colony_mask.hdf5"

n_labels = 2
autoencoder = cla_seg(n_labels)
autoencoder.load_weights(weight_file)


# In[6]:


def bg_correction(image, order=1):
    """returns corrected image based on poly_matrix and np.linalg.lstsq

    Parameters
    ----------
    image :
        2d image, with height=width
    order : int, optional
        the order of poly matrix

    Returns
    ----------
    2d corrected_image
    """

    def poly_matrix(_x_grid_flattened, _y_grid_flattened, _order):
        """generate a poly matrix"""
        ncols = (_order + 1) ** 2
        mat = np.zeros((_x_grid_flattened.size, ncols))
        # cartesian_product_indices = itertools.product(range(_order + 1), range(_order + 1))
        cartesian_product_indices = [(i, j) for i in range(_order + 1) for j in range(_order + 1)]
        for k, (i, j) in enumerate(cartesian_product_indices):
            mat[:, k] = _x_grid_flattened**i * _y_grid_flattened**j
        return mat

    assert image.shape[0] == image.shape[1], "image shape: h is not equal to w"
    x_range, y_range = np.arange(0, image.shape[0], 1), np.arange(0, image.shape[0], 1)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    poly_mat = poly_matrix(x_grid.flatten(), y_grid.flatten(), order)

    # Solve for np.dot(poly_mat, lstsq_sol) = z_sol:
    lstsq_sol = np.linalg.lstsq(poly_mat, image.flatten())[0]

    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # new_poly_mat = poly_matrix(x_grid.ravel(), y_grid.ravel(), order)
    new_poly_mat = poly_matrix(x_grid.flatten(), y_grid.flatten(), order)

    z_sol = np.reshape(np.dot(new_poly_mat, lstsq_sol), x_grid.shape)
    z_mean = np.mean(z_sol)

    # TODO: add an argument to do bg correction directly
    # zz_min=np.amin(zz)
    # bg=zz-np.amin(zz)
    # return Img-zz+np.amin(zz)
    return z_sol


# In[7]:


def prep_prediction_data(img_path, img_list):
    data = []
    i = 0
    img0 = np.array(imread(img_path + img_list[0], dtype=np.float64))
    bg0, bg0_mean = bg_correction(img0, order=2)
    bg = bg0 - bg0_mean
    while i < len(img_list):
        img = np.array(imread(img_path + img_list[i]), dtype=np.float64)
        img = img - bg
        img = (img - np.amin(img)) * 1.0 / (np.amax(img) - np.amin(img))  # img*1.0 transform array to double
        img = img * 1.0 / np.median(img)
        img_h = img.shape[0]
        img_w = img.shape[1]
        img = np.reshape(img, (img_h, img_w, 1))
        data.append(img)
        i += 1
    data = np.array(data)
    return data


# In[ ]:


for posi in range(1, posi_end + 1):
    img_path = input_path + str(posi) + "/"
    img_list = sorted(listdir(img_path))
    posi_path = output_path + str(posi) + "/colony_mask/"
    if not os.path.exists(posi_path):
        os.makedirs(posi_path)
    predict_data = prep_prediction_data(img_path, img_list)
    print(predict_data.shape)
    output = autoencoder.predict(predict_data, batch_size=1, verbose=0)
    for i in range(output.shape[0]):
        im = np.argmax(output[i], axis=-1)
        # save image to the exat value
        img_name = "mask_" + img_list[i][0 : len(img_list[i]) - 4]
        img = Image.fromarray(im.astype(np.uint32), "I")
        img.save(posi_path + img_name + ".png")


# In[ ]:
