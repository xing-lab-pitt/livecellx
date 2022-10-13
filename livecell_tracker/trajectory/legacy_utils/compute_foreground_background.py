#!/usr/bin/env python
# coding: utf-8

# In[1]:


"1 calculate foreground and background by calculate mean values of the whole experiment2 use B-spline to get smooth function3 The compensate value are defined by max(fg_value)-fg and max(bg_value)-bg"


# In[37]:


import glob
import os
import pickle
from math import pi
from os import listdir

import cv2
import numpy as np
import pandas as pd
import scipy.misc
import scipy.ndimage as ndi
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import bisplev, bisplrep
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import h_maxima, local_maxima, remove_small_objects
from skimage.segmentation import clear_border, watershed

# In[38]:


main_path = "/home/zoro/Desktop/experiment_data/2019-05-17_A549_vim/"

fluor_path = main_path + "/vimentin/"
fluor_output_path = main_path + "output/"

posi_end = 20
fluor_step = 12
sample_step = 32
img_h, img_w = 1952, 1952


# In[39]:


# ----------get mean background and mean foreground
tot_fg = np.zeros((img_h, img_w))
tot_fg_mask = (tot_fg > 0).astype(np.float64)

tot_bg = np.zeros((img_h, img_w))
tot_bg_mask = (tot_bg > 0).astype(np.float64)

for posi in range(1, posi_end + 1):
    print(posi)
    fluor_img_path = fluor_path + str(posi) + "/"
    fluor_seg_path = fluor_output_path + str(posi) + "/seg/"
    fluor_img_list = sorted(listdir(fluor_img_path))
    fluor_seg_list = sorted(listdir(fluor_seg_path))

    for i in np.arange(0, len(fluor_img_list), fluor_step):
        img_num = i + 1
        fluor_img = imread(fluor_img_path + fluor_img_list[i])
        seg_img = imread(fluor_seg_path + fluor_seg_list[i])

        cur_mask = seg_img > 0
        tot_fg = fluor_img * cur_mask + tot_fg
        tot_fg_mask += cur_mask.astype(np.float64)

        cur_bg_mask = seg_img == 0
        tot_bg = fluor_img * cur_bg_mask + tot_bg
        tot_bg_mask += cur_bg_mask.astype(np.float64)


# In[40]:


plt.imshow(tot_fg)
plt.show()
plt.imshow(tot_fg_mask)
plt.show()
print(np.amax(tot_fg), np.amax(tot_fg_mask), np.amin(tot_fg_mask))
mean_fg = tot_fg / (tot_fg_mask + 1e-10)
plt.imshow(mean_fg)
plt.show()
# np.save(main_path+'mean_fg.npy',mean_fg)
# ---------use gaussian filter to avoid local fluctuation, Do not use large filters
mean_fg_blur = gaussian(mean_fg, 48, preserve_range=True)
plt.imshow(mean_fg_blur)
plt.show()


# In[41]:


plt.imshow(tot_bg)
plt.show()
plt.imshow(tot_bg_mask)
plt.show()
print(np.amax(tot_bg), np.amax(tot_bg_mask), np.amin(tot_bg_mask))
mean_bg = tot_bg / (tot_bg_mask + 1e-10)
plt.imshow(mean_bg)
plt.show()
# np.save(main_path+'mean_bg.npy',mean_bg)


# In[42]:


# mean_fg=tot_fg/(tot_fg_mask+1e-10)
# mean_bg=tot_bg/(tot_bg_mask+1e-10)


# In[43]:


I = mean_fg_blur
n_row, n_col = I.shape[0], I.shape[1]
ctrl_x = []
ctrl_y = []
ctrl_z = []


for i in np.arange(0, n_row, sample_step):
    for j in np.arange(0, n_col, sample_step):
        if tot_fg_mask[i, j] > 0:  # !!!!!!!!!!!!!based on condition
            ctrl_x.append(i)
            ctrl_y.append(j)
            ctrl_z.append(I[i, j])
ctrl_x = np.array(ctrl_x)
print(ctrl_x.shape)
ctrl_y = np.array(ctrl_y)
ctrl_z = np.array(ctrl_z)

fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.scatter(ctrl_x, ctrl_y, ctrl_z)
plt.show()

nx, ny = (1952, 1952)
lx = np.linspace(0, n_row, nx)
ly = np.linspace(0, n_col, ny)


# s value is important for smoothing
tck = bisplrep(ctrl_x, ctrl_y, ctrl_z, s=1e10)
znew = bisplev(lx, ly, tck)
fig = plt.figure()
ax = fig.gca(projection="3d")
x, y = np.meshgrid(lx, ly)
surf = ax.plot_surface(x, y, znew, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

plt.imshow(znew)
plt.show()
np.save(main_path + "foreground_vimentin_smooth.npy", znew)


fg_offset = znew - np.amin(znew)  # np.amax(znew)-znew#
plt.imshow(fg_offset)
plt.show()


fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(x, y, fg_offset, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()


# In[ ]:


# In[44]:


I = mean_bg
n_row, n_col = I.shape[0], I.shape[1]
ctrl_x = []
ctrl_y = []
ctrl_z = []


for i in np.arange(0, n_row, sample_step):
    for j in np.arange(0, n_col, sample_step):
        if tot_bg_mask[i, j] > 0:
            ctrl_x.append(i)
            ctrl_y.append(j)
            ctrl_z.append(I[i, j])
ctrl_x = np.array(ctrl_x)
ctrl_y = np.array(ctrl_y)
ctrl_z = np.array(ctrl_z)

fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.scatter(ctrl_x, ctrl_y, ctrl_z)
plt.show()

nx, ny = I.shape[0], I.shape[1]
lx = np.linspace(0, n_row, nx)
ly = np.linspace(0, n_col, ny)


# s value is important for smoothing
tck = bisplrep(ctrl_x, ctrl_y, ctrl_z, s=1e10)
znew = bisplev(lx, ly, tck)
fig = plt.figure()
ax = fig.gca(projection="3d")
x, y = np.meshgrid(lx, ly)
surf = ax.plot_surface(x, y, znew, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

plt.imshow(znew)
plt.show()

np.save(main_path + "background_vimentin_smooth.npy", znew)


bg_offset = znew - np.amin(znew)  # np.amax(znew)-znew#
plt.imshow(bg_offset)
plt.show()

fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(x, y, bg_offset, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()


# In[ ]:


# In[46]:


np.save(main_path + "background_offset_vimentin", bg_offset)
np.save(main_path + "foreground_offset_vimentin", fg_offset)


# In[ ]:


# In[ ]:
