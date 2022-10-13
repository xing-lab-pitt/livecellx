#!/usr/bin/env python
# coding: utf-8

# In[12]:


import glob
import itertools
import os
import sys
from os import listdir

import cv2
import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Activation, BatchNormalization, Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D
from keras.models import Model
from matplotlib import pyplot as plt
from PIL import Image
from legacy_utils.reg_seg_model import reg_seg
from scipy import signal
from scipy.ndimage import distance_transform_edt, filters
from skimage.exposure import equalize_adapthist
from skimage.io import imread, imread_collection

# In[7]:


main_path = "/home/zoro/Desktop/experiment_data/2018-05-18_HK2_2d_traj/"
input_path = main_path + "img/"
output_path = main_path + "output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

posi_end = 22

weight_file = "HK2_reg_aug.hdf5"
autoencoder = reg_seg()
autoencoder.load_weights(weight_file)


# In[8]:


# ---------------dispose shadow on camera------------
Img_shadow = imread(main_path + "/shadow.tif")
shadow_mask = Img_shadow == 0


# In[9]:


def bg_correction(Img, ordr=1):
    def poly_matrix(x, y, order=1):
        """ generate Matrix use with lstsq """
        ncols = (order + 1) ** 2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order + 1), range(order + 1))
        for k, (i, j) in enumerate(ij):
            G[:, k] = x ** i * y ** j
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
    # zz_min=np.amin(zz)
    zz_mean = np.mean(zz)
    # bg=zz-np.amin(zz)
    # return Img-zz+np.amin(zz)
    return zz, zz_mean


# In[10]:


def prep_prediction_data(img_path, img_list):
    data = []
    i = 0
    img0 = np.array(imread(img_path + img_list[0], dtype=np.float64))
    bg0, bg0_mean = bg_correction(img0, ordr=2)
    bg = bg0 * shadow_mask + Img_shadow - bg0_mean
    while i < len(img_list):
        img = imread(img_path + img_list[i])
        img = img - bg

        # --------------!!!! equalize_adapthist--------------
        img = img.astype(np.uint16)
        img = equalize_adapthist(img)

        img = (img - np.amin(img)) * 1.0 / (np.amax(img) - np.amin(img))  # img*1.0 transform array to double
        img = img * 1.0 / np.median(img)
        img_h = img.shape[0]
        img_w = img.shape[1]
        img = np.reshape(img, (img_h, img_w, 1))
        data.append(img)
        i += 1
    data = np.array(data)
    return data


# In[13]:


for posi in range(1, posi_end + 1):
    img_path = input_path + str(posi) + "/"
    img_list = sorted(listdir(img_path))
    posi_path = output_path + str(posi) + "/reg/"
    if not os.path.exists(posi_path):
        os.makedirs(posi_path)
    predict_data = prep_prediction_data(img_path, img_list)
    print(predict_data.shape)
    output = autoencoder.predict(predict_data, batch_size=1, verbose=0)
    for i in range(output.shape[0]):
        im = output[i][:, :, 0]
        # save image to the exat value
        img = Image.fromarray(im)
        img.save(posi_path + "reg_" + img_list[i])


# In[ ]:
