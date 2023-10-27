import os
import pickle
from math import exp, log
from os import listdir

import contour_class
import image_warp
import numpy as np
import pandas as pd
import scipy.interpolate.fitpack as fitpack
import scipy.ndimage as ndimage
import utils
from cell_class import fluor_single_cell, single_cell
from contour_tool import (
    align_contour_to,
    align_contours,
    df_find_contour_points,
    find_contour_points,
    generate_contours,
)
from matplotlib import pyplot as plt

# import sparse
from scipy import signal
from scipy.signal import medfilt, wiener
from scipy.stats import kde
from skimage import measure
from skimage.io import imread
from skimage.morphology import closing, opening
from skimage.segmentation import find_boundaries
from sklearn import (
    cluster,
    decomposition,
    manifold,
    metrics,
    mixture,
    model_selection,
    preprocessing,
    random_projection,
)
from sklearn.neighbors import BallTree, kneighbors_graph
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from statsmodels import robust
from traj_class import fluor_single_cell_traj, single_cell_traj

key_list = [
    "Cell_AreaShape_Area",
    "Cell_AreaShape_BoundingBoxArea",
    "Cell_AreaShape_BoundingBoxMaximum_X",
    "Cell_AreaShape_BoundingBoxMaximum_Y",
    "Cell_AreaShape_BoundingBoxMinimum_X",
    "Cell_AreaShape_BoundingBoxMinimum_Y",
    "Cell_AreaShape_Center_X",
    "Cell_AreaShape_Center_Y",
    "Cell_AreaShape_Compactness",
    "Cell_AreaShape_Eccentricity",
    "Cell_AreaShape_EquivalentDiameter",
    "Cell_AreaShape_EulerNumber",
    "Cell_AreaShape_Extent",
    "Cell_AreaShape_FormFactor",
    "Cell_AreaShape_MajorAxisLength",
    "Cell_AreaShape_MaxFeretDiameter",
    "Cell_AreaShape_MaximumRadius",
    "Cell_AreaShape_MeanRadius",
    "Cell_AreaShape_MedianRadius",
    "Cell_AreaShape_MinFeretDiameter",
    "Cell_AreaShape_MinorAxisLength",
    "Cell_AreaShape_Orientation",
    "Cell_AreaShape_Perimeter",
    "Cell_AreaShape_Solidity",
]
key_mask = np.ones((len(key_list),), dtype=bool)
key_mask[key_list.index("Cell_AreaShape_BoundingBoxArea")] = False
key_mask[key_list.index("Cell_AreaShape_BoundingBoxMaximum_X")] = False
key_mask[key_list.index("Cell_AreaShape_BoundingBoxMaximum_Y")] = False
key_mask[key_list.index("Cell_AreaShape_BoundingBoxMinimum_X")] = False
key_mask[key_list.index("Cell_AreaShape_BoundingBoxMinimum_Y")] = False
key_mask[key_list.index("Cell_AreaShape_Center_Y")] = False
key_mask[key_list.index("Cell_AreaShape_Center_X")] = False
key_mask[key_list.index("Cell_AreaShape_EquivalentDiameter")] = False
key_mask[key_list.index("Cell_AreaShape_EulerNumber")] = False
key_mask[key_list.index("Cell_AreaShape_Orientation")] = False

# --------scaling by time point with smallest variance--------
def cal_sma(X, window_size=24):  # sliding mean
    sma = [np.mean(X[i : i + window_size, :], axis=0) for i in range(X.shape[0] - window_size)]
    sma = np.array(sma)
    return sma


def slide_std(X, window_size=None):  # sliding standard deviation
    sstd = [np.std(X[i : i + window_size, :], axis=0) for i in range(X.shape[0] - window_size)]
    sstd = np.array(sstd)
    return sstd


def slide_mad(X, window_size=24):  # sliding Median Absolute Deviation
    smad = [robust.scale.mad(X[i : i + window_size, :], c=1, axis=0) for i in range(X.shape[0] - window_size)]
    smad = np.array(smad)
    return smad


def traj_scaling(sct, wd_size=12, t_range=36, sort_range=6):

    mask = sct.traj_vimentin_feature_values[0] != 0
    traj_t = sct.traj_seri[mask][:, 0]
    traj_cord = sct.traj_cord[mask]
    traj_vim = sct.traj_vimentin_haralick_pca_cord[mask]

    morph_sma = cal_sma(traj_cord, window_size=wd_size)

    morph_sstd = slide_std(traj_cord, window_size=wd_size)

    #     morph_smad=slide_mad(traj_cord,window_size=wd_size)

    sort_morph_std_ind = np.argsort(np.sum(morph_sstd[:t_range, :], axis=1))[:sort_range]

    #     print(np.amin(morph_sma[sort_morph_std_ind,0]))

    morph_scale_t = sort_morph_std_ind[np.argmin(morph_sma[sort_morph_std_ind, 0])]

    morph_scale_area = cal_sma(sct.traj_feature[mask][:, key_mask], window_size=wd_size)[morph_scale_t, 0]
    morph_scale_ar = cal_sma(sct.traj_feature[mask][:, key_mask], window_size=wd_size)[morph_scale_t, 8]  # mean redius
    morph_scale_mr = cal_sma(sct.traj_feature[mask][:, key_mask], window_size=wd_size)[
        morph_scale_t, 9
    ]  # median redius

    scale_contour = sct.traj_contour[mask] / np.sqrt(morph_scale_area)

    vim_mi_sma = cal_sma(sct.traj_vimentin_feature_values[0][mask][:, None], window_size=wd_size)
    vim_sma = cal_sma(sct.traj_vimentin_feature_values[2][mask], window_size=wd_size)
    vim_sstd = slide_std(sct.traj_vimentin_feature_values[2][mask], window_size=wd_size)
    sort_vim_std_ind = np.argsort(np.sum(vim_sstd[:t_range, :], axis=1))[:sort_range]
    vim_scale_t = sort_vim_std_ind[np.argmin(vim_mi_sma[sort_vim_std_ind, 0])]

    vim_scale = vim_sma[vim_scale_t, :]
    print(morph_scale_t, vim_scale_t)
    scale_haralick = sct.traj_vimentin_feature_values[2][mask] - vim_scale

    return scale_contour, scale_haralick


# --------scaling by the first stay point--------
# https://github.com/Yurui-Li/Stay-Point-Identification
def GetDistance(data, ind1, ind2, Metric_L):  # Metric_L=1: Manhattan distance , 2: Euclidean distance
    distance = np.linalg.norm(data[ind2] - data[ind1], ord=Metric_L)
    return distance


def ka(dis, dc):  # dc cutoff distance
    if dis >= dc:
        return 0
    else:
        return 1


# local density
def density(data, dc, Metric_L):
    part_density = []  # local density
    scope = []  # density range
    leftBoundary = 0
    rightBoundary = len(data) - 1
    for i in range(len(data)):
        traigger = True
        left = i - 1
        right = i + 1
        incrementLeft = 1
        incrementRight = 1
        while traigger:
            # extend left
            if incrementLeft != 0:
                if left < 0:
                    left = leftBoundary
                distanceLeft = GetDistance(data, left, i, Metric_L=Metric_L)
                if (distanceLeft < dc) & (left > leftBoundary):
                    left -= 1
                else:
                    incrementLeft = 0
            # extend right
            if incrementRight != 0:
                if right > rightBoundary:
                    right = rightBoundary
                distanceRight = GetDistance(data, i, right, Metric_L=Metric_L)
                if (distanceRight < dc) & (right < rightBoundary):
                    right += 1
                else:
                    incrementRight = 0
            # stop extend
            if (incrementLeft == 0) & (incrementRight == 0):
                traigger = False
            if (left == leftBoundary) & (incrementRight == 0):
                traigger = False
            if (incrementLeft == 0) & (right == rightBoundary):
                traigger = False
        if left == leftBoundary:
            scope.append([left, right - 1])
            part_density.append(right - left - 1)
        elif right == rightBoundary:
            scope.append([left + 1, right])
            part_density.append(right - left - 1)
        else:
            scope.append([left + 1, right - 1])
            part_density.append(right - left - 2)
    part_density = np.array(part_density)
    scope = np.array(scope)
    return part_density, scope


# reverse update
def SP_search(data, part_density, scope, tc, Metric_L):
    SP = []

    traigger = True
    used = []
    while traigger:
        partD = max(part_density)
        index = np.argmax(part_density)
        #         print('index:',index)
        start = scope[index][0]
        end = scope[index][1]

        if len(used) != 0:
            for i in used:
                if (scope[i][0] > start) & (scope[i][0] < end):
                    part_density[index] = scope[i][0] - start - 1
                    scope[index][1] = scope[i][0] - 1
                #                     print("1_1")
                if (scope[i][1] > start) & (scope[i][1] < end):
                    part_density[index] = end - scope[i][1] - 1
                    scope[index][0] = scope[i][1] + 1
                #                     print("1_2")
                if (scope[i][0] <= start) & (scope[i][1] >= end):
                    part_density[index] = 0
                    scope[index][0] = 0
                    scope[index][1] = 0
            #                     print("1_3")
            start = scope[index][0]
            end = scope[index][1]
        timeCross = end - start
        #         print('time:',timeCross)
        if timeCross > tc:
            S_arrive_t = start
            S_leave_t = end

            SP.append(index)
            used.append(index)
            for k in range(scope[index][0], scope[index][1] + 1):
                part_density[k] = 0
        part_density[index] = 0
        if max(part_density) == 0:
            traigger = False
    SP = np.array(SP)
    return SP


# judge stay points overlap
def similar(sp, data, dc, Metric_L):
    index = sp.tolist()
    redundant = []
    for i in index:
        for j in index:
            if i not in redundant and j > i:
                dist = GetDistance(data, i, j, Metric_L=Metric_L)
                if dist < dc:
                    redundant.append(j)
    print(index, redundant)
    for k in set(redundant):
        index.remove(k)
    index = np.array(index)
    return index


def sp_traj_scaling(sct, t_cutoff=6, t_range=48, Metric_L=1, norm_flag=False):
    mask = sct.traj_vimentin_feature_values[0] != 0
    traj_t = sct.traj_seri[mask][:, 0]
    traj_morph = sct.traj_morph_pca_cord[mask]
    if norm_flag == True:
        traj_vim = sct.traj_norm_vimentin_haralick_pca_cord[mask]
    else:
        traj_vim = sct.traj_vimentin_haralick_pca_cord[mask]

    # morph_scaler=StandardScaler().fit(traj_morph[:,:3].flatten()[:,None])
    # vim_scaler=StandardScaler().fit(traj_vim[:,:3].flatten()[:,None])

    morph_scaler = MinMaxScaler().fit(traj_morph[:, :].flatten()[:, None])
    vim_scaler = MinMaxScaler().fit(traj_vim[:, :].flatten()[:, None])

    norm_traj_morph = traj_morph.copy()
    for i in range(traj_morph.shape[1]):
        norm_traj_morph[:, i] = morph_scaler.transform(traj_morph[:, i][:, None])[:, 0]
    norm_traj_vim = traj_vim.copy()
    for i in range(traj_vim.shape[1]):
        norm_traj_vim[:, i] = vim_scaler.transform(traj_vim[:, i][:, None])[:, 0]

    X = np.column_stack((norm_traj_morph, norm_traj_vim))

    dot_color = np.arange(X.shape[0])
    cm = plt.cm.get_cmap("jet")
    #     plt.scatter(norm_traj_morph[:,0],norm_traj_morph[:,1],c=dot_color,cmap=cm,s=1)
    #     plt.show()
    #     plt.scatter(norm_traj_vim[:,0],norm_traj_vim[:,1],c=dot_color,cmap=cm,s=1)
    #     plt.show()
    plt.scatter(norm_traj_morph[:, 0], norm_traj_vim[:, 0], c=dot_color, cmap=cm, s=1)
    #     plt.show()

    dist_cutoff = max(
        np.mean(np.linalg.norm(np.diff(X[:t_range], axis=0), axis=1, ord=Metric_L)),
        np.mean(np.linalg.norm(np.diff(X, axis=0), axis=1, ord=Metric_L)),
    )
    #     dist_cutoff=np.mean(np.linalg.norm(np.diff(X,axis=0),axis=1))
    print(dist_cutoff)
    part_density, scope = density(X, dist_cutoff, Metric_L=Metric_L)
    SP = SP_search(X, part_density, scope, t_cutoff, Metric_L=Metric_L)
    #     print(SP)
    if SP.shape[0] > 0 and np.amin(SP) < t_range:

        scale_t = np.amin(SP)

        #     scale_t=np.argmax(part_density[:48])
        print(scale_t, scope[scale_t])

        plt.scatter(norm_traj_morph[SP, 0], norm_traj_vim[SP, 0])
        plt.show()

        if scope[scale_t][1] - scope[scale_t][0] < 2 * t_cutoff:
            st = min(max(0, scale_t - t_cutoff), scope[scale_t][0])
            et = max(scale_t + t_cutoff, scope[scale_t][1])
        else:
            st, et = scope[scale_t][0], scope[scale_t][1]
        print(st, et)

        morph_scale_area = np.mean(
            sct.traj_feature[mask][:, key_mask][st:et, 0]
        )  # key map sets the features, 0 index sets the specific feature (area)
        morph_scale_ar = np.mean(sct.traj_feature[mask][:, key_mask][st:et, 8])  # mean redius
        morph_scale_mr = np.mean(sct.traj_feature[mask][:, key_mask][st:et, 9])  # median redius

        scale_contour = sct.traj_contour / np.sqrt(morph_scale_area)
        scale_contour_with_vim = sct.traj_contour[mask] / np.sqrt(morph_scale_area)

        if norm_flag == True:
            ind = 4
        else:
            ind = 3

        vim_scale = np.mean(sct.traj_vimentin_feature_values[ind][mask][st:et, :], axis=0)

        scale_haralick = sct.traj_vimentin_feature_values[ind][mask] - vim_scale

    else:
        if SP.shape[0] > 0:
            plt.scatter(norm_traj_morph[SP, 0], norm_traj_vim[SP, 0])
            plt.show()
        scale_contour, scale_contour_with_vim, scale_haralick, scale_t = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    return scale_contour, scale_contour_with_vim, scale_haralick, scale_t


# #calculate stay points separately
def ssp_traj_scaling(sct, t_cutoff=6, t_range=48, Metric_L=1, norm_flag=False):
    mask = sct.traj_vimentin_feature_values[0] != 0
    traj_t = sct.traj_seri[mask][:, 0]
    traj_morph = sct.traj_cord[mask]
    if norm_flag == True:
        traj_vim = sct.traj_norm_vimentin_haralick_pca_cord[mask]
    else:
        traj_vim = sct.traj_vimentin_haralick_pca_cord[mask]

    X = np.column_stack((traj_morph, traj_vim))

    dot_color = np.arange(X.shape[0])
    cm = plt.cm.get_cmap("jet")

    morph_dist_cutoff = max(
        np.mean(np.linalg.norm(np.diff(traj_morph[:t_range], axis=0), axis=1, ord=Metric_L)),
        np.mean(np.linalg.norm(np.diff(traj_morph, axis=0), axis=1, ord=Metric_L)),
    )
    vim_dist_cutoff = max(
        np.mean(np.linalg.norm(np.diff(traj_vim[:t_range], axis=0), axis=1, ord=Metric_L)),
        np.mean(np.linalg.norm(np.diff(traj_vim, axis=0), axis=1, ord=Metric_L)),
    )
    #     dist_cutoff=np.mean(np.linalg.norm(np.diff(X,axis=0),axis=1))
    print(morph_dist_cutoff, vim_dist_cutoff)
    morph_part_density, morph_scope = density(traj_morph, morph_dist_cutoff, Metric_L=Metric_L)
    morph_SP = SP_search(traj_morph, morph_part_density, morph_scope, t_cutoff, Metric_L=Metric_L)

    vim_part_density, vim_scope = density(traj_vim, vim_dist_cutoff, Metric_L=Metric_L)
    vim_SP = SP_search(traj_vim, vim_part_density, vim_scope, t_cutoff, Metric_L=Metric_L)
    print(morph_SP, vim_SP)
    if morph_SP.shape[0] > 0 and vim_SP.shape[0] > 0 and np.amin(morph_SP) < t_range and np.amin(vim_SP) < t_range:

        morph_scale_t = np.amin(morph_SP)
        vim_scale_t = np.amin(vim_SP)

        #     scale_t=np.argmax(part_density[:48])
        print(morph_scale_t, morph_scope[morph_scale_t])
        print(vim_scale_t, vim_scope[vim_scale_t])

        plt.scatter(traj_morph[:, 0], traj_morph[:, 1], c=dot_color, cmap=cm, s=1)
        plt.scatter(traj_morph[morph_SP, 0], traj_morph[morph_SP, 1])
        plt.show()
        plt.scatter(traj_vim[:, 0], traj_vim[:, 1], c=dot_color, cmap=cm, s=1)
        plt.scatter(traj_vim[vim_SP, 0], traj_vim[vim_SP, 1])
        plt.show()

        # if scope[scale_t][1]-scope[scale_t][0]<2*t_cutoff:
        #     st=min(max(0,scale_t-t_cutoff),scope[scale_t][0])
        #     et=max(scale_t+t_cutoff,scope[scale_t][1])
        # else:
        #     st,et=scope[scale_t][0],scope[scale_t][1]
        # print(st,et)
        morph_st, morph_et = morph_scope[morph_scale_t][0], morph_scope[morph_scale_t][1]
        vim_st, vim_et = vim_scope[vim_scale_t][0], vim_scope[vim_scale_t][1]

        morph_scale_area = np.mean(sct.traj_feature[mask][:, key_mask][morph_st:morph_et, 0])
        morph_scale_ar = np.mean(sct.traj_feature[mask][:, key_mask][morph_st:morph_et, 8])  # mean redius
        morph_scale_mr = np.mean(sct.traj_feature[mask][:, key_mask][morph_st:morph_et, 9])  # median redius

        scale_contour = sct.traj_contour / np.sqrt(morph_scale_area)
        scale_contour_with_vim = sct.traj_contour[mask] / np.sqrt(morph_scale_area)

        if norm_flag == True:
            ind = 4
        else:
            ind = 3

        vim_scale = np.mean(sct.traj_vimentin_feature_values[ind][mask][vim_st:vim_et, :], axis=0)

        scale_haralick = sct.traj_vimentin_feature_values[ind][mask] - vim_scale
    else:
        scale_contour, scale_contour_with_vim, scale_haralick, morph_scale_t, vim_scale_t = [], [], [], [], []

    return scale_contour, scale_contour_with_vim, scale_haralick, morph_scale_t, vim_scale_t
