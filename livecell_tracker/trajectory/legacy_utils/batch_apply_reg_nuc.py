#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import sys
from os import listdir

import cv2
import numpy as np
import pandas as pd
from .cnn_prep_data import prep_fluor_data
from keras import optimizers
from keras.layers import Activation, BatchNormalization, Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D
from keras.models import Model
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from .reg_seg_model import reg_seg
from scipy import signal
from scipy.interpolate import bisplev, bisplrep
from scipy.ndimage import distance_transform_edt, filters
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.io import imread, imread_collection

# In[2]:


main_path = "/home/zoro/Desktop/experiment_data/2019-03-10_HK2_fucci/"
input_path = main_path + "cdt1/"
output_path = main_path + "cdt1_output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

posi_end = 20

weight_file = "HK2_nuc.hdf5"
autoencoder = reg_seg()
autoencoder.load_weights(weight_file)


# In[30]:


for posi in range(1, posi_end + 1):
    img_path = input_path + str(posi) + "/"
    img_list = sorted(listdir(img_path))
    posi_path = output_path + str(posi) + "/reg/"
    if not os.path.exists(posi_path):
        os.makedirs(posi_path)
    for i in range(len(img_list)):
        img_num = i + 1
        predict_data = prep_fluor_data(img_path, img_list, img_num)
        output = autoencoder.predict(predict_data, batch_size=1, verbose=0)
        # save image to the exat value
        img = Image.fromarray(output[0, :, :, 0])
        img.save(posi_path + "reg_" + img_list[i])


# In[ ]:
