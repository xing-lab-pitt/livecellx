#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir

import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import closing, dilation, disk, opening
from skimage.segmentation import find_boundaries

# In[2]:


main_path = "/home/zoro/Desktop/experiment_data/ATCC data/"
posi_end = 20
input_path = main_path + "test_img/"
# input_path='/media/zoro/easystore/xing-lab-1/experiment_data/2019-03-10_HK2_fucci/cdt1/'
output_path = main_path + "output/"


# In[4]:


def find_contours_labelimg(seg_img, contour_value):
    seg_img = opening(seg_img)
    contours = []

    rps = regionprops(seg_img)
    r_labels = [r.label for r in rps]
    # print(r_labels)
    contours = []
    for label in r_labels:
        single_obj_seg_img = seg_img == label
        single_contour = find_contours(
            single_obj_seg_img, level=contour_value, fully_connected="low", positive_orientation="low"
        )
        # print(len(single_contour))
        max_len = 0
        for i in range(len(single_contour)):
            if len(single_contour[i]) >= max_len:
                maj_i = i
                max_len = len(single_contour[i])
        # need append the element of in single_contour instead of the whole
        # array
        contours.append(single_contour[maj_i])
    return contours


# In[6]:


# for posi in range(1,posi_end+1):
#     print(posi)
#     img_path=input_path+str(posi)+'/'
#     seg_path=output_path+str(posi)+'/seg/'
img_path = input_path
seg_path = output_path + "seg/"
img_list = sorted(listdir(img_path))
seg_list = sorted(listdir(seg_path))


img_num = 1


img = imread(img_path + img_list[img_num - 1])
seg = imread(seg_path + seg_list[img_num - 1])
#     seg=dilation(seg,disk(9))
contours = find_contours_labelimg(seg, contour_value=0.5)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)
plt.show()
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img, interpolation="nearest", cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis("image")
plt.show()
