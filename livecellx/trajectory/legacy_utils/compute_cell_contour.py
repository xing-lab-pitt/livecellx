#!/usr/bin/env python
# coding: utf-8

# In[9]:


import copy
import pickle
from os import listdir

import legacy_utils.contour_class as contour_class
import legacy_utils.image_warp as image_warp
import numpy as np
import pandas as pd
import scipy.interpolate.fitpack as fitpack
import scipy.ndimage as ndimage
import seaborn as sns
import legacy_utils.utils as utils
from legacy_utils.config import *
from legacy_utils.contour_tool import (
    align_contour_to,
    align_contours,
    df_find_contour_points,
    find_contour_points,
    generate_contours,
)
from matplotlib import pyplot as plt
from scipy.stats import kde
from skimage import measure
from skimage.io import imread
from skimage.morphology import closing, opening
from skimage.segmentation import find_boundaries

# In[4]:


# main_path='/home/zoro/Desktop/experiment_data/2019-03-22_a549_tgf4ng_2d/'
# main_path = '/mnt/data0/Ke/weikang_exp_process/2019-05-05_A549_vim/'
# sample_path='/home/zoro/Desktop/experiment_data/2019-03-22_a549_tgf4ng_2d/output/2/seg/'
# img_list=sorted(listdir(sample_path))
# sample_img_list=[]
# for i in np.arange(0,len(img_list),24):
#     sample_img_list.append(img_list[i])
# print(len(sample_img_list))


# In[10]:


# sample_path='/home/zoro/Desktop/experiment_data/2019-03-22_a549_tgf4ng_2d/seg_sample2/'
sample_img_list = sorted(listdir(sample_path))
print(sample_img_list)


# In[11]:


contour_points_and_obj = find_contour_points(sample_path, sample_img_list, contour_value=0.5)


# In[12]:


cell_contours, sort_obj_arr = generate_contours(
    contour_points_and_obj, closed_only=True, min_area=None, max_area=None, axis_align=True
)

print("contour generation complete")
# In[13]:
for i in range(len(cell_contours)):
    cell_contours[i].resample(num_points=150)
    cell_contours[i].axis_align()
    points = cell_contours[i].points
    plt.plot(points[:, 0], points[:, 1], ".")
plt.show()


# In[14]:


# be careful about the allow_scaling
mean_contour, iters = align_contours(cell_contours, allow_reflection=True, allow_scaling=False, max_iters=20)

print("contour align complete")
# In[14]:


print(iters)
# with open(main_path+'output/A549_ctrl_mean_cell_contour', 'wb') as fp:
with open(main_path + "output/mean_cell_contour", "wb") as fp:
    pickle.dump(mean_contour, fp)


# In[19]:


# with open(main_path+'/A549_ctrl_mean_cell_contour', 'rb') as fp:
#     mean_contour=pickle.load(fp)
# with open(main_path+'/A549_emt_mean_cell_contour', 'rb') as fp:
#     emt_mean_contour=pickle.load(fp)
plt.plot(mean_contour.points[:, 0], mean_contour.points[:, 1], ".")
# plt.plot(emt_mean_contour.points[:, 0], emt_mean_contour.points[:, 1], '.')
# plt.legend(('control','TGF-beta'))

plt.show()


# In[15]:


for i in range(len(cell_contours)):
    points = cell_contours[i].points
    plt.plot(points[:, 0], points[:, 1], ".")
plt.show()


# In[16]:


for i in range(len(cell_contours)):
    scale_back = utils.decompose_homogenous_transform(cell_contours[i].to_world_transform)[1]
    cell_contours[i].scale(scale_back)
    points = cell_contours[i].points
    plt.plot(points[:, 0], points[:, 1], ".")
plt.show()


# In[17]:


pca_contours = contour_class.PCAContour.from_contours(
    contours=cell_contours, required_variance_explained=0.98, return_positions=False
)


# In[14]:


with open(main_path + "/pca_contours", "wb") as fp:
    pickle.dump(pca_contours, fp)

# with open (sample_path+'/pca_contours', 'rb') as fp:
#     pca_contours = pickle.load(fp)


# In[18]:


# -------plot principal modes-------------------------
for pci in range(pca_contours.position.shape[0]):
    cell_posi = pca_contours.position
    mode_std1 = copy.copy(cell_posi)
    mode_std1[pci] = 1
    mode_std2 = copy.copy(cell_posi)
    mode_std2[pci] = 2
    mode_std_1 = copy.copy(cell_posi)
    mode_std_1[pci] = -1
    mode_std_2 = copy.copy(cell_posi)
    mode_std_2[pci] = -2

    print(cell_posi)
    print(mode_std1)
    print(mode_std2)
    print(mode_std_1)
    print(mode_std_2)
    shape_array0 = contour_class.PCAContour.points_at_position(pca_contours, cell_posi)
    shape_array1 = contour_class.PCAContour.points_at_position(pca_contours, mode_std1)
    shape_array2 = contour_class.PCAContour.points_at_position(pca_contours, mode_std2)
    shape_array3 = contour_class.PCAContour.points_at_position(pca_contours, mode_std_1)
    shape_array4 = contour_class.PCAContour.points_at_position(pca_contours, mode_std_2)

    for i in [0, 1, 3]:
        com_str = "points = shape_array" + str(i)
        print(com_str)
        exec(com_str)
        #     print(points.shape)
        # fig, ax = plt.subplots(figsize=(12, 12))

        plt.plot(points[:, 0], points[:, 1], "-", linewidth=6)
        plt.legend(("mean", "+1", "-1"))

        plt.axis("equal")
        plt.axis([-120, 120, -70, 70])
    plt.show()


# In[14]:


# cell_posi=np.array([ -1.,  -1. , 0.,  0. , 0. , 0. , 0.])
# points=contour_class.PCAContour.points_at_position(pca_contours,cell_posi)
# plt.plot(points[:, 0], points[:, 1], '-',linewidth=6)


# plt.axis('equal')
# plt.axis([-120,120,-70,70])
# plt.show()


# In[19]:


all_cord = []
for i in range(len(cell_contours)):
    cell_posi = contour_class.PCAContour.find_position(pca_contours, cell_contours[i])
    all_cord.append(cell_posi)
    plt.scatter(cell_posi[0], cell_posi[1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
plt.savefig("mean_contour1.png")
plt.close()

# In[20]:


all_cord = np.array(all_cord)

x, y = all_cord[:, 0], all_cord[:, 1]
sns.kdeplot(x, y, n_levels=200, shade=True)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
plt.savefig("mean_contour2.png")
plt.close()


nbins = 300
k = kde.gaussian_kde([x, y])
xi, yi = np.mgrid[x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# Make the plot
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.jet)
plt.show()
plt.savefig("mean_contour3.png")
plt.close()
# In[ ]:


# In[ ]:
