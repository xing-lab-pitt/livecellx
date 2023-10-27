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
from contour_tool import (
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

main_path = "/mnt/data0/Ke/weikang_exp_process/2019-05-05_A549_vim/"
# with open (main_path+'/pca_contours', 'rb') as fp:
#     pca_contours = pickle.load(fp)


with open(main_path + "/output/mean_cell_contour", "rb") as fp:
    mean_contour = pickle.load(fp)
plt.plot(mean_contour.points[:, 0], mean_contour.points[:, 1], ".")

plt.savefig("mean_contour0.png")
