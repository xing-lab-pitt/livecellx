import itertools
import math
import pickle
import time
from math import pi
from os import listdir

import cv2
import numpy as np
import pandas as pd
import scipy
from legacy_utils.index import Indexes
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from legacy_utils.resnet50 import res_model
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from skimage import morphology
from skimage.color import label2rgb
from skimage.exposure import equalize_adapthist
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

# this is faster than the one from lineage mapper,only with matrix calculation


def compute_overlap_matrix(img_path, img_list, img_num_1, img_num_2):
    frame_1 = imread(img_path + "/" + img_list[img_num_1 - 1])
    frame_2 = imread(img_path + "/" + img_list[img_num_2 - 1])
    nb_cell_1 = np.amax(frame_1)
    nb_cell_2 = np.amax(frame_2)

    frame_overlap = np.zeros((nb_cell_1, nb_cell_2))
    for obj_idx1 in range(nb_cell_1):
        obj_num1 = obj_idx1 + 1
        sc_img = frame_1 == obj_num1
        ol_judge = np.logical_and(sc_img, frame_2)
        ol_value = np.multiply(ol_judge, frame_2)
        ol_obj2 = np.unique(ol_value).tolist()
        # ol_obj2=ol_obj2[ol_obj2!=0]
        ol_obj2.remove(0)
        if len(ol_obj2) > 0:
            for obj_num2 in ol_obj2:
                ol_area = np.sum(ol_value == obj_num2)
                obj_idx2 = obj_num2 - 1
                frame_overlap[obj_idx1][obj_idx2] = ol_area

    return frame_overlap


# compute relative overlap of two objects in different images


def compute_overlap_pair(img_path, img_list, img_num_1, obj_num_1, img_num_2, obj_num_2):
    frame_1 = imread(img_path + "/" + img_list[img_num_1 - 1])
    frame_2 = imread(img_path + "/" + img_list[img_num_2 - 1])
    nb_cell_1 = np.amax(frame_1)
    nb_cell_2 = np.amax(frame_2)

    target_overlap = np.zeros(nb_cell_2)
    source_overlap = np.zeros(nb_cell_1)

    sc_img1 = frame_1 == obj_num_1
    obj1_area = np.sum(sc_img1)
    ol_judge = np.logical_and(sc_img1, frame_2)
    ol_value = np.multiply(ol_judge, frame_2)
    tar_obj_list = np.unique(ol_value).tolist()
    tar_obj_list.remove(0)
    for tar_obj in tar_obj_list:
        ol_area = np.sum(ol_value == tar_obj)
        tar_idx = tar_obj - 1
        target_overlap[tar_idx] = ol_area / obj1_area

    sc_img2 = frame_2 == obj_num_2
    obj2_area = np.sum(sc_img2)
    ol_judge = np.logical_and(sc_img2, frame_1)
    ol_value = np.multiply(ol_judge, frame_1)
    sou_obj_list = np.unique(ol_value).tolist()
    sou_obj_list.remove(0)
    for sou_obj in sou_obj_list:
        ol_area = np.sum(ol_value == sou_obj)
        sou_idx = sou_obj - 1
        source_overlap[sou_idx] = ol_area / obj2_area

    return target_overlap, source_overlap


# compute relative overlap of 1 objects in another image
# compute overlap of an object in another image
def compute_overlap_single(img_path, img_list, img_num_1, obj_num_1, img_num_2):
    frame_1 = imread(img_path + "/" + img_list[img_num_1 - 1])
    frame_2 = imread(img_path + "/" + img_list[img_num_2 - 1])
    nb_cell_1 = np.amax(frame_1)
    nb_cell_2 = np.amax(frame_2)

    target_overlap = np.zeros(nb_cell_2)

    sc_img1 = frame_1 == obj_num_1
    obj_area = np.sum(sc_img1)

    ol_judge = np.logical_and(sc_img1, frame_2)
    ol_value = np.multiply(ol_judge, frame_2)
    tar_obj_list = np.unique(ol_value).tolist()
    tar_obj_list.remove(0)
    for tar_obj in tar_obj_list:
        ol_area = np.sum(ol_value == tar_obj)
        tar_idx = tar_obj - 1
        target_overlap[tar_idx] = ol_area / obj_area
    return target_overlap


# for relabel all traj based on traj start time
def relabel_traj(df):
    traj_labels = df["Cell_TrackObjects_Label"].values
    traj_labels = np.sort(np.unique(traj_labels[traj_labels > 0]))
    traj_quantity = len(traj_labels)  # the quantity of trajectories
    print("traj quantity", traj_quantity)
    # -------------relabel all traj from 1,sort base on start time of each traj
    traj_label_st = []  # record traj start time and traj label
    for i in range(traj_quantity):
        cur_traj_label = traj_labels[i]
        traj_st = np.asscalar(df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, "ImageNumber"].values[0])
        traj_label_st.append([traj_st, cur_traj_label])
    traj_label_st = np.array(traj_label_st)
    traj_label_st = traj_label_st[traj_label_st[:, 0].argsort()]
    # -------------relabel must be careful because the dataframe is modified every loop
    # --------------use a copy to modify by index
    df_relabel = df.copy()
    for traj_i in range(traj_quantity):
        cur_traj_label = traj_label_st[traj_i, 1]
        cur_traj_inds = df[df["Cell_TrackObjects_Label"] == cur_traj_label].index
        df_relabel.loc[cur_traj_inds, "Cell_TrackObjects_Label"] = int(traj_i + 1)
    return df_relabel


def generate_traj_df(df):
    """
    Record img_num and obj_num(or idx_num in Per_Object) in all traj into one table, label=rowIndex+1
    Returns
        a tuple of dataframe
    """
    t_span = max(df["ImageNumber"])
    traj_label = df["Cell_TrackObjects_Label"].values
    traj_label = np.sort(np.unique(traj_label[traj_label > 0]))
    num_trajectories = len(traj_label)  # the quantity of trajectories
    print("#trajectories:", num_trajectories)

    t_col = [str(i + 1) for i in range(t_span)]

    # initialize pandas dataframes
    traj_df = -1 * np.ones((num_trajectories, t_span), dtype=np.int)
    traj_df = pd.DataFrame(traj_df, columns=t_col)

    traj_to_row_idx_df = -1 * np.ones((num_trajectories, t_span), dtype=np.int)
    traj_to_row_idx_df = pd.DataFrame(traj_to_row_idx_df, columns=t_col)

    for traj_i in range(num_trajectories):
        cur_traj_label = traj_label[traj_i]
        # find all the index that have the same label(in the same trajectory)
        same_traj_label_indices = df["Cell_TrackObjects_Label"] == int(cur_traj_label)
        row_idx_list = df[same_traj_label_indices].index.tolist()
        for row_idx in row_idx_list:
            time_index = df["ImageNumber"][row_idx]
            traj_df[str(time_index)][traj_i] = df["ObjectNumber"][row_idx]
            traj_to_row_idx_df[str(time_index)][traj_i] = row_idx
    return traj_df, traj_to_row_idx_df


# return numpy record of each traj start and traj end. [img_num obj_num]
def record_traj_start_end(df):
    traj_labels = df["Cell_TrackObjects_Label"].values
    traj_labels = np.sort(np.unique(traj_labels[traj_labels > 0]))
    traj_quan = len(traj_labels)  # the quantity of trajectories
    print("traj quantity", traj_quan)

    traj_start = []
    traj_end = []
    for cur_traj_label in traj_labels:
        traj_start.append(
            df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, ["ImageNumber", "ObjectNumber"]].values[0].tolist()
        )
        traj_end.append(
            df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, ["ImageNumber", "ObjectNumber"]].values[-1].tolist()
        )
    traj_start = np.array(traj_start)
    traj_end = np.array(traj_end)
    return traj_start, traj_end


# judge Max OverLap type
def judge_mol_type(frame_overlap, source_obj, target_obj):  # obj max_overlap relation judge
    rel_flag = 0
    # print(source_obj.shape, source_obj.min(), source_obj.max(), source_obj)
    to_ol_child = frame_overlap[source_obj - 1, :]
    if np.all(to_ol_child == 0):  # if source obj have no overlap child
        max_child_obj = 0
    else:
        max_child_obj = np.argmax(frame_overlap, axis=1)[source_obj - 1] + 1

    so_ol_parent = frame_overlap[:, target_obj - 1]
    if np.all(so_ol_parent == 0):  # if target obj have no overlap parent
        max_parent_obj = 0
    else:
        max_parent_obj = np.argmax(frame_overlap, axis=0)[target_obj - 1] + 1

    if source_obj != max_parent_obj and max_child_obj == target_obj:
        rel_flag = 1

    if source_obj == max_parent_obj and max_child_obj != target_obj:
        rel_flag = 2

    if source_obj != max_parent_obj and max_child_obj != target_obj:
        rel_flag = 3
    return rel_flag


def search_false_link(df, relation_df, frame_overlap, img_num_1, source_obj, img_num_2, target_obj, rel_flag):
    break_pair = []
    to_ol_child = frame_overlap[source_obj - 1, :]
    if np.all(to_ol_child == 0):  # if source obj have no overlap child
        max_child_obj = 0
    else:
        max_child_obj = np.argmax(frame_overlap, axis=1)[source_obj - 1] + 1

    so_ol_parent = frame_overlap[:, target_obj - 1]
    if np.all(so_ol_parent == 0):  # if target obj have no overlap parent
        max_parent_obj = 0
    else:
        max_parent_obj = np.argmax(frame_overlap, axis=0)[target_obj - 1] + 1

    # rel_flag=1:source_obj!=max_parent_obj and max_child_obj==target_obj
    # include:
    # 1 oversegmentation at img_num_1
    # 2 undersegmentation at img_num_2
    # 3 cell that moves a lot but with a little ovelap between parent and child
    # when rel_flag=1, max_parent_obj cannot be 0, because target obj already
    # overlap with source obj
    if rel_flag == 1:
        max_parent_rel_child = relation_df.loc[
            (relation_df["image_number1"] == img_num_1) & (relation_df["object_number1"] == max_parent_obj),
            "object_number2",
        ].values
        if max_parent_rel_child.size == 0:
            break_pair = [img_num_1, source_obj, img_num_2, target_obj]

    # rel_flag=2:source_obj==max_parent_obj and max_child_obj!=target_obj:
    # include:
    # 1 under-segmentation at img_num_1
    # 2 over-seg at img_num_2
    # 3 apoptosis or mitosis cell that ovelap max with wrong mother
    # max_child_obj cannot be 0, because source obj already overlap with target

    if rel_flag == 2:
        max_child_rel_parent = relation_df.loc[
            (relation_df["image_number2"] == img_num_2) & (relation_df["object_number2"] == max_child_obj),
            "object_number1",
        ].values
        if max_child_rel_parent.size == 0:
            break_pair = [img_num_1, source_obj, img_num_2, target_obj]

    # rel_flag=3:source_obj!=max_parent_obj and max_child_obj!=target_obj
    # include
    # 1 mis-assignment
    # 2 cell moves a lot without any overlap between parent and child(this is
    # right link)

    if rel_flag == 3:
        if max_parent_obj != 0 and max_child_obj == 0:
            max_parent_rel_child = relation_df.loc[
                (relation_df["image_number1"] == img_num_1) & (relation_df["object_number1"] == max_parent_obj),
                "object_number2",
            ].values

            if max_parent_rel_child.size == 0:
                # print('max_child_obj=0','overseg broken')
                break_pair = [img_num_1, source_obj, img_num_2, target_obj]
            if max_parent_rel_child.size == 1:
                max_parent_rel_child = int(np.asscalar(max_parent_rel_child))
                flag3_1 = judge_mol_type(frame_overlap, max_parent_obj, max_parent_rel_child)
                # print('max_child_obj=0',flag1,max_parent_obj,max_parent_rel_child)
                if flag3_1 == 3:
                    break_pair = [img_num_1, source_obj, img_num_2, target_obj]

        elif max_parent_obj == 0 and max_child_obj != 0:
            max_child_rel_parent = relation_df.loc[
                (relation_df["image_number2"] == img_num_2) & (relation_df["object_number2"] == max_child_obj),
                "object_number1",
            ].values
            if max_child_rel_parent.size == 0:
                # print('max_parent_obj=0,','traj_transfer')
                break_pair = [img_num_1, source_obj, img_num_2, target_obj]
            if max_child_rel_parent.size == 1:
                max_child_rel_parent = int(np.asscalar(max_child_rel_parent))
                flag3_2 = judge_mol_type(frame_overlap, max_child_rel_parent, max_child_obj)
                # print('max_parent_obj=0',flag2,max_child_rel_parent,max_child_obj)
                if flag3_2 == 3:
                    break_pair = [img_num_1, source_obj, img_num_2, target_obj]

        #         elif max_parent_obj==0 and max_child_obj==0:
        #             print('cell move quickly in low density')

        elif max_parent_obj != 0 and max_child_obj != 0:
            flag_broken = 0
            flag_transfer = 0
            flag3_3 = 0
            flag3_4 = 0
            max_parent_rel_child = relation_df.loc[
                (relation_df["image_number1"] == img_num_1) & (relation_df["object_number1"] == max_parent_obj),
                "object_number2",
            ].values

            if max_parent_rel_child.size == 0:
                # print('target_associate_pair,','overseg broken')
                flag_broken = 1
            if max_parent_rel_child.size == 1:
                max_parent_rel_child = int(np.asscalar(max_parent_rel_child))
                flag3_3 = judge_mol_type(frame_overlap, max_parent_obj, max_parent_rel_child)
                # print('target_associate_pair',flag3,max_parent_obj,max_parent_rel_child)

            max_child_rel_parent = relation_df.loc[
                (relation_df["image_number2"] == img_num_2) & (relation_df["object_number2"] == max_child_obj),
                "object_number1",
            ].values
            if max_child_rel_parent.size == 0:
                # print('source_associate_pair,','traj_transfer')
                flag_transfer = 1
            if max_child_rel_parent.size == 1:
                max_child_rel_parent = int(np.asscalar(max_child_rel_parent))
                flag3_4 = judge_mol_type(frame_overlap, max_child_rel_parent, max_child_obj)
                # print('source_associate_pair',flag4,max_child_rel_parent,max_child_obj)

            if flag_broken == 1 or flag_transfer == 1 or flag3_3 == 3 or flag3_4 == 3 or (flag3_3 > 0 and flag3_4 > 0):
                break_pair = [img_num_1, source_obj, img_num_2, target_obj]

    return break_pair


# def find_border_obj(img_path,img_list,img_num):
#     border_obj=[]
#     img=imread(img_path+'/'+img_list[img_num-1])
#     clear_border_img=clear_border(img)
#     border_img=img-clear_border_img
#     border_obj=np.unique(border_img).tolist()
#     border_obj.remove(0)
#     return border_obj
# border_obj=find_border_obj(seg_path,seg_img_list,1)


def find_border_obj(img_path, img_list, img_num):
    border_obj = []
    img = imread(img_path + "/" + img_list[img_num - 1])
    img_h = img.shape[0]
    img_w = img.shape[1]
    rps = regionprops(img)
    r_labels = [r.label for r in rps]
    r_bboxes = [r.bbox for r in rps]
    for i in range(len(r_labels)):
        obj_bbox = r_bboxes[i]
        if obj_bbox[0] == 0 or obj_bbox[1] == 0 or obj_bbox[2] == img_h or obj_bbox[3] == img_w:
            border_obj.append([img_num, r_labels[i]])
    return border_obj


def judge_border(img_path, img_list, img_num, obj_num):
    border_flag = 0
    img = imread(img_path + "/" + img_list[img_num - 1])
    img_h = img.shape[0]
    img_w = img.shape[1]
    rps = regionprops(img)
    r_labels = [r.label for r in rps]
    obj_i = r_labels.index(obj_num)
    obj_bbox = [r.bbox for r in rps][obj_i]
    if obj_bbox[0] == 0 or obj_bbox[1] == 0 or obj_bbox[2] == img_h or obj_bbox[3] == img_w:
        border_flag = 1
    return border_flag


def break_link(df, relation_df, false_link):
    max_label = max(df["Cell_TrackObjects_Label"])
    for img_num_1, source_obj, img_num_2, target_obj in false_link:
        target_idx = np.asscalar(df[(df["ImageNumber"] == img_num_2) & (df["ObjectNumber"] == target_obj)].index.values)
        target_label = df.loc[target_idx, "Cell_TrackObjects_Label"]

        if df.loc[target_idx, "Cell_TrackObjects_LinkType"] > 0:
            max_label += 1
            # print(img_num_1,source_obj,img_num_2,target_obj,max_label)
            df.at[target_idx, "Cell_TrackObjects_ParentImageNumber"] = 0
            df.at[target_idx, "Cell_TrackObjects_ParentObjectNumber"] = 0
            df.at[target_idx, "Cell_TrackObjects_LinkType"] = 0
            df.at[
                (df["Cell_TrackObjects_Label"] == target_label) & (df["ImageNumber"] >= img_num_2),
                "Cell_TrackObjects_Label",
            ] = max_label

            relation_df = relation_df.drop(
                relation_df[
                    (relation_df["image_number1"] == img_num_1) & (relation_df["object_number1"] == source_obj)
                ].index
            )
    return df, relation_df


# mark false_seg_object 'Cell_TrackObjects_Label' as -1
def false_seg_mark(df, false_seg_obj):
    for img_num, obj_num in false_seg_obj:
        df.loc[(df["ImageNumber"] == img_num) & (df["ObjectNumber"] == obj_num), "Cell_TrackObjects_Label"] = -1
    return df


# connect link and re-order all traj labels


def connect_link(df, relation_df, pairs_need_link):
    for parent_i_n, parent_o_n, child_i_n, child_o_n in pairs_need_link:
        child_label = np.asscalar(
            df.loc[
                (df["ImageNumber"] == child_i_n) & (df["ObjectNumber"] == child_o_n), "Cell_TrackObjects_Label"
            ].values
        )
        # ---------parent label should from the direct parent objects,because traj label may be updated(traj relink 2 or more times)
        parent_label = np.asscalar(
            df.loc[
                (df["ImageNumber"] == parent_i_n) & (df["ObjectNumber"] == parent_o_n), "Cell_TrackObjects_Label"
            ].values
        )
        child_start_idx = df[(df["ImageNumber"] == child_i_n) & (df["ObjectNumber"] == child_o_n)].index.values
        df.at[child_start_idx, "Cell_TrackObjects_Label"] = parent_label
        df.at[child_start_idx, "Cell_TrackObjects_ParentImageNumber"] = parent_i_n
        df.at[child_start_idx, "Cell_TrackObjects_ParentObjectNumber"] = parent_o_n
        df.at[child_start_idx, "Cell_TrackObjects_LinkType"] = 1
        df.loc[df["Cell_TrackObjects_Label"] == child_label, "Cell_TrackObjects_Label"] = parent_label

        new_rel = pd.DataFrame(
            [[1, parent_i_n, parent_o_n, child_i_n, child_o_n]],
            columns=["relationship_type_id", "image_number1", "object_number1", "image_number2", "object_number2"],
        )
        # print(new_rel)
        relation_df = relation_df.append(new_rel, ignore_index=True)
        # print(relation_df.shape)

    return df, relation_df


# for judging the traj start part or end part is in am_record(apoptosis or
# mitosis)
def judge_traj_am(df, am_record, img_num, obj_num, judge_later=True, t_range=3):
    t_span = max(df["ImageNumber"])
    obj_am_flag = 0
    obj_traj_label = np.asscalar(
        df.loc[(df["ImageNumber"] == img_num) & (df["ObjectNumber"] == obj_num)]["Cell_TrackObjects_Label"].values
    )
    if judge_later:
        ot_piece = df.loc[
            (df["ImageNumber"] >= img_num)
            & (df["ImageNumber"] < min(t_span, img_num + t_range))
            & (df["Cell_TrackObjects_Label"] == obj_traj_label),
            ["ImageNumber", "ObjectNumber"],
        ].values.tolist()
    else:
        ot_piece = df.loc[
            (df["ImageNumber"] <= img_num)
            & (df["ImageNumber"] > max(0, img_num - t_range))
            & (df["Cell_TrackObjects_Label"] == obj_traj_label),
            ["ImageNumber", "ObjectNumber"],
        ].values.tolist()
    # will return the am_flag of the last time point in t_range, the last time
    # point reflect more possible state:mitosis to apoptosis
    for ot_i_n, ot_o_n in ot_piece:
        if ((am_record["ImageNumber"] == ot_i_n) & (am_record["ObjectNumber"] == ot_o_n)).any():
            obj_am_flag = np.asscalar(
                am_record.loc[
                    (am_record["ImageNumber"] == ot_i_n) & (am_record["ObjectNumber"] == ot_o_n), "am_flag"
                ].values
            )
    return obj_am_flag


def judge_apoptosis_tracklet(df, am_record, img_num, obj_num):
    apo_flag = 0
    cur_traj_label = np.asscalar(
        df.loc[(df["ImageNumber"] == img_num) & (df["ObjectNumber"] == obj_num)]["Cell_TrackObjects_Label"].values
    )
    cur_traj_end = df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, ["ImageNumber", "ObjectNumber"]].values[-1]

    ot_size_arr = df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, "Cell_AreaShape_Area"].values
    ot_length = len(ot_size_arr)
    ot_mean_size = np.mean(ot_size_arr)
    sort_ot_size_arr = np.sort(ot_size_arr)
    if ot_length <= 3:
        # start_am_flag=judge_traj_am(df,am_record,img_num,obj_num,judge_later=True,t_range=1)
        end_am_flag = judge_traj_am(df, am_record, cur_traj_end[0], cur_traj_end[1], judge_later=False, t_range=1)
        if end_am_flag == 1:
            apo_flag = 1
    if ot_length > 3 and ot_length <= 5:
        # start_am_flag=judge_traj_am(df,am_record,img_num,obj_num,judge_later=True,t_range=2)
        end_am_flag = judge_traj_am(df, am_record, cur_traj_end[0], cur_traj_end[1], judge_later=False, t_range=2)
        if end_am_flag == 1:
            apo_flag = 1
    if ot_length > 5 and ot_length <= 10:
        # start_am_flag=judge_traj_am(df,am_record,img_num,obj_num,judge_later=True,t_range=3)
        end_am_flag = judge_traj_am(df, am_record, cur_traj_end[0], cur_traj_end[1], judge_later=False, t_range=3)
        if end_am_flag == 1:
            apo_flag = 1
    if ot_length > 10:
        start_am_flag = judge_traj_am(df, am_record, img_num, obj_num, judge_later=True, t_range=5)
        end_am_flag = judge_traj_am(df, am_record, cur_traj_end[0], cur_traj_end[1], judge_later=False, t_range=5)
        size_variation = np.mean(sort_ot_size_arr[ot_length - 5 : ot_length]) / np.mean(sort_ot_size_arr[:5])
        if start_am_flag == 1 and end_am_flag == 1 and size_variation < 2:
            apo_flag = 1

    return apo_flag


# def img_transform(img,obj_h,obj_w):
#     img=(img-np.amin(img))*1.0/(np.amax(img)-np.amin(img))#img*1.0 transform array to double
#     img=img*1.0/np.median(img)
#     img=cv2.resize(img, (obj_h,obj_w), interpolation=cv2.INTER_CUBIC)
#     img=np.reshape(img,(obj_h,obj_w,1))
#     return img

# def generate_single_seg_img(img_path,seg_path,img_list,seg_img_list,obj_h,obj_w,img_num,obj_num):
#     img=imread(img_path+'/'+img_list[img_num-1])
#     seg_img=imread(seg_path+'/'+seg_img_list[img_num-1])

#     #single_obj_img=morphology.binary_dilation(seg_img==obj_num,morphology.diamond(16))
#     single_obj_img=seg_img==obj_num
#     single_obj_img=label(single_obj_img)
#     rps=regionprops(single_obj_img)
#     candi_r=[r for r in rps if r.label==1][0]
#     candi_box=candi_r.bbox
#     single_cell_img=single_obj_img*img
#     crop_img=single_cell_img[candi_box[0]:candi_box[2],candi_box[1]:candi_box[3]]
#     #-----------equalize_adapthist-----------------------------
#     crop_img =equalize_adapthist(crop_img,nbins=100)


#     inds=ndimage.distance_transform_edt(crop_img==0, return_distances=False, return_indices=True)
#     crop_img=crop_img[tuple(inds)]
#     crop_img=img_transform(crop_img,obj_h,obj_w)
#     crop_img=np.reshape(crop_img,(1,obj_h,obj_w,1))
#     return crop_img

# return numpy record of each traj start and traj end. [img_num obj_num]
def traj_start_end_info(df):
    traj_labels = df["Cell_TrackObjects_Label"].values
    traj_labels = np.sort(np.unique(traj_labels[traj_labels > 0]))
    traj_quan = len(traj_labels)  # the quantity of trajectories
    print("traj quantity", traj_quan)

    traj_start = []
    traj_start_xy = []
    traj_start_area = []
    traj_end = []
    traj_end_xy = []
    traj_end_area = []
    for cur_traj_label in traj_labels:
        traj_start.append(
            df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, ["ImageNumber", "ObjectNumber"]].values[0].tolist()
        )
        traj_end.append(
            df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, ["ImageNumber", "ObjectNumber"]].values[-1].tolist()
        )
        traj_start_xy.append(
            df.loc[
                df["Cell_TrackObjects_Label"] == cur_traj_label, ["Cell_AreaShape_Center_X", "Cell_AreaShape_Center_Y"]
            ]
            .values[0]
            .tolist()
        )
        traj_end_xy.append(
            df.loc[
                df["Cell_TrackObjects_Label"] == cur_traj_label, ["Cell_AreaShape_Center_X", "Cell_AreaShape_Center_Y"]
            ]
            .values[-1]
            .tolist()
        )
        traj_start_area.append(
            df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, "Cell_AreaShape_Area"].values[0].tolist()
        )
        traj_end_area.append(
            df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, "Cell_AreaShape_Area"].values[-1].tolist()
        )
    traj_start = np.array(traj_start)
    traj_end = np.array(traj_end)
    traj_start_xy = np.array(traj_start_xy)
    traj_end_xy = np.array(traj_end_xy)
    traj_start_area = np.array(traj_start_area)
    traj_end_area = np.array(traj_end_area)
    return traj_start, traj_end, traj_start_xy, traj_end_xy, traj_start_area, traj_end_area


def am_obj_info(am_record, df):
    am_arr = []
    am_area = []
    am_xy = []

    for index, row in am_record.iterrows():
        am_i_n = row["ImageNumber"]
        am_o_n = row["ObjectNumber"]
        am_arr.append([am_i_n, am_o_n])
        # am_flag=row['am_flag']
        am_area.append(
            df.loc[(df["ImageNumber"] == am_i_n) & (df["ObjectNumber"] == am_o_n), "Cell_AreaShape_Area"]
            .values[0]
            .tolist()
        )
        am_xy.append(
            df.loc[
                (df["ImageNumber"] == am_i_n) & (df["ObjectNumber"] == am_o_n),
                ["Cell_AreaShape_Center_X", "Cell_AreaShape_Center_Y"],
            ]
            .values[0]
            .tolist()
        )

    am_arr = np.array(am_arr)
    am_xy = np.array(am_xy)
    am_area = np.array(am_area)
    return am_arr, am_xy, am_area


def compute_specific_overlap(img_path, img_list, img_num_1, img_num_2, obj_num_arr1, obj_num_arr2):
    frame_1 = imread(img_path + "/" + img_list[img_num_1 - 1])
    frame_2 = imread(img_path + "/" + img_list[img_num_2 - 1])

    frame_overlap = np.zeros((len(obj_num_arr1), len(obj_num_arr2)))
    for i in range(obj_num_arr1.shape[0]):
        obj_num1 = obj_num_arr1[i]
        sc_img = frame_1 == obj_num1
        ol_judge = np.logical_and(sc_img, frame_2)
        ol_value = np.multiply(ol_judge, frame_2)
        ol_obj2 = np.unique(ol_value).tolist()
        ol_obj2.remove(0)
        if len(ol_obj2) > 0:
            for obj_num2 in ol_obj2:
                if np.any(obj_num_arr2 == obj_num2):
                    j = np.where(obj_num_arr2 == obj_num2)
                    ol_area = np.sum(ol_value == obj_num2)
                    frame_overlap[i][j] = ol_area
    return frame_overlap


def compute_cost(
    frame_overlap,
    centroids_1,
    centroids_2,
    cells_size_1,
    cells_size_2,
    weight_overlap=1,
    weight_centroids=1,
    weight_size=0.5,
    max_centroids_distance=150,
):

    number_cells_1 = len(cells_size_1)
    number_cells_2 = len(cells_size_2)
    # Expand input vectors to matrices
    M_centroids_1_X = np.tile(centroids_1[:, 0], (number_cells_2, 1)).T
    M_centroids_1_Y = np.tile(centroids_1[:, 1], (number_cells_2, 1)).T
    M_centroids_2_X = np.tile(centroids_2[:, 0], (number_cells_1, 1))
    M_centroids_2_Y = np.tile(centroids_2[:, 1], (number_cells_1, 1))
    M_cells_size_1 = np.tile(cells_size_1, (number_cells_2, 1)).T
    M_cells_size_2 = np.tile(cells_size_2, (number_cells_1, 1))

    # Compute the centroid_term. delta_centroid is the distance between a
    # source cell centroid and a target cell centroid.
    centroid_term = (
        np.sqrt((M_centroids_1_X - M_centroids_2_X) ** 2 + (M_centroids_1_Y - M_centroids_2_Y) ** 2)
        / max_centroids_distance
    )

    # Compute the overlap_term
    overlap_term = 1 - (frame_overlap / (2 * M_cells_size_1) + frame_overlap / (2 * M_cells_size_2))

    # Compute the size_term
    size_term = np.absolute(M_cells_size_1 - M_cells_size_2) / np.maximum(M_cells_size_1, M_cells_size_2)
    # print(centroid_term,overlap_term,size_term)
    # Compute the cost
    frame_cost = weight_overlap * overlap_term + weight_centroids * centroid_term + weight_size * size_term

    # Set to nan the invalid track costs
    frame_cost[centroid_term > 1] = 1e100
    # frame_cost[overlap_term==1]=1e100
    return frame_cost


def cal_am_link_pairs(img_num, seg_path, seg_img_list):
    am_link_pairs = []

    img_num_1 = img_num
    img_num_2 = img_num + 1
    mask1 = am_arr[:, 0] == img_num_1
    mask2 = am_arr[:, 0] == img_num_2
    o_n_arr1 = am_arr[mask1][:, 1]
    o_n_arr2 = am_arr[mask2][:, 1]
    xy_arr1 = am_xy[mask1]
    xy_arr2 = am_xy[mask2]
    area_arr1 = am_area[mask1]
    area_arr2 = am_area[mask2]
    frame_overlap = compute_specific_overlap(seg_path, seg_img_list, img_num_1, img_num_2, o_n_arr1, o_n_arr2)
    frame_cost = compute_cost(frame_overlap, xy_arr1, xy_arr2, area_arr1, area_arr2)
    row_ind, col_ind = linear_sum_assignment(frame_cost)
    for r, c in zip(row_ind, col_ind):
        if frame_cost[r, c] != 1e100:
            am_link_pairs.append([img_num_1, o_n_arr1[r], img_num_2, o_n_arr2[c]])
    return am_link_pairs


def find_am_sisters(F, mitosis_max_distance=50, size_simi_thres=0.7):
    """Compute scores for matching a parent to two daughters

    F - an N x 5 (or more) array x,y,img_num,obj_num,area


    """

    X = 0

    Y = 1

    IIDX = 2  # img_num

    OIDX = 3  # obj_num

    AIDX = 4  # area

    if len(F) <= 1:

        return np.array([])

    max_distance = mitosis_max_distance

    # Find all daughter pairs within same frame

    i, j = np.where(F[:, np.newaxis, IIDX] == F[np.newaxis, :, IIDX])

    i, j = i[i < j], j[i < j]  # get rid of duplicates and self-compares

    dmax = max_distance * 2 - np.sqrt(np.sum((F[i, :2] - F[j, :2]) ** 2, 1))

    dist_mask = dmax >= 0

    i, j, dmax = i[dist_mask], j[dist_mask], dmax[dist_mask]

    size_simi = 1 - abs((F[i, AIDX] - F[j, AIDX]) / (F[i, AIDX] + F[j, AIDX]))
    size_mask = np.zeros(i.shape, dtype=bool)

    uni_t = np.unique(F[i, IIDX])
    ii = np.arange(len(i))
    for ti in uni_t:
        match_inds = []
        mask_ti = F[i[ii], IIDX] == ti
        ti_size_simi = size_simi[ii[mask_ti]]
        sort_inds = np.argsort(-ti_size_simi)
        for si in sort_inds:
            if ti_size_simi[si] < size_simi_thres:
                break
            ii_where = np.where(size_simi == ti_size_simi[si])[0]
            i_where = i[ii_where][0]
            j_where = j[ii_where][0]
            if i_where in match_inds or j_where in match_inds:
                continue
            else:
                match_inds.append(i_where)
                match_inds.append(j_where)
                size_mask[ii_where] = True

    i, j, dmax = i[size_mask], j[size_mask], dmax[size_mask]
    return np.column_stack((F[i, 2:4], F[j, 2:4]))


# -------calculate the cell fusion -----------------------


def cal_cell_fusion(frame_overlap, img_num_1, img_num_2, nb_cell_1, nb_cell_2):
    prefuse_group = (
        []
    )  # each element is a list include all prefuse cells in a fuse event, corresponding to postfuse_cells
    postfuse_cells = []  # include: img_num,obj_num
    frame_fusion = np.zeros(frame_overlap.shape)
    for source_o_n in range(1, nb_cell_1 + 1):
        # find target whose max_overlap mother is source
        ol_target = frame_overlap[source_o_n - 1, :]
        if np.all(ol_target == 0):  # if source obj have no overlap target
            target_o_n = 0
        else:
            # axis=1,maximum of each row,return column index
            target_o_n = np.argmax(frame_overlap, axis=1)[source_o_n - 1] + 1

        if target_o_n > 0:
            frame_fusion[source_o_n - 1, target_o_n - 1] = 1

        # Compute the sum vector S which is the sum of all the columns of frame_fusion matrix. The fusion target region
        # will have at least 2 cells tracked to it => S>1
    S = np.sum(frame_fusion, axis=0)
    frame_fusion[:, S == 1] = 0
    # Update the sum vector
    S = np.sum(frame_fusion, axis=0)

    for i in range(len(np.where(S >= 2)[0])):
        f_group = []
        # num of prefuse cells:S[np.where(S >= 2)[0][i]]
        postfuse_cells.append([img_num_2, np.where(S >= 2)[0][i] + 1])
        frame_fusion_i = frame_fusion[:, np.where(S >= 2)[0][i]]

        for r in range(len(np.where(frame_fusion_i == 1)[0])):
            # fuse_pairs.append([img_num_1,np.where(frame_fusion_i==1)[0][r]+1,img_num_2,np.where(S >= 2)[0][i]+1])
            f_group.append([img_num_1, np.where(frame_fusion_i == 1)[0][r] + 1])
        prefuse_group.append(f_group)
    return postfuse_cells, prefuse_group


# ------------------------------calculate cell split --------------------------
# transpose the frame_overlap matrix, use the algorithm used in
# fusion,reverse time order to calculate split


def cal_cell_split(frame_overlap, img_num_1, img_num_2, nb_cell_1, nb_cell_2):
    presplit_cells = []  # include: img_num,obj_num
    postsplit_group = []  # each element is a list include all postsplit cells in a split event

    # row is cells in frame2, column is cells in frame1
    frame_overlap_R = np.transpose(frame_overlap)
    frame_split = np.zeros(frame_overlap_R.shape)
    for source_R_o_n in range(1, nb_cell_2 + 1):

        ol_target_R = frame_overlap_R[source_R_o_n - 1, :]
        if np.all(ol_target_R == 0):  # if source_R obj have no overlap target_R
            target_R_o_n = 0
        else:
            target_R_o_n = np.argmax(ol_target_R) + 1

        if target_R_o_n > 0:
            frame_split[source_R_o_n - 1, target_R_o_n - 1] = 1

    # Compute the sum vector S which is the sum of all the columns of frame_split matrix. The split target_R region
    # will have at least 2 cells tracked to it => S>1
    S = np.sum(frame_split, axis=0)
    frame_split[:, S == 1] = 0

    # Update the sum vector
    S = np.sum(frame_split, axis=0)

    for i in range(len(np.where(S >= 2)[0])):
        s_group = []
        # number_postsplit cells:S[np.where(S >= 2)[0][i]]
        presplit_cells.append([img_num_1, np.where(S >= 2)[0][i] + 1])
        frame_split_i = frame_split[:, np.where(S >= 2)[0][i]]
        for r in range(len(np.where(frame_split_i == 1)[0])):
            # split_pairs.append([img_num_1,np.where(S >= 2)[0][i]+1,img_num_2,np.where(frame_split_i==1)[0][r]+1])
            s_group.append([img_num_2, np.where(frame_split_i == 1)[0][r] + 1])
        postsplit_group.append(s_group)
    return presplit_cells, postsplit_group


def find_mitosis_pairs_to_break(relation_df, candi_am_sisters, false_link):
    mitosis_pairs_to_break = []
    nm_ind = []
    for i in range(len(candi_am_sisters)):
        mis_link = 0
        rel_pairs = []

        sis1_i_n = candi_am_sisters[i][0]
        sis1_o_n = candi_am_sisters[i][1]
        sis2_i_n = candi_am_sisters[i][2]
        sis2_o_n = candi_am_sisters[i][3]

        if ((relation_df["image_number2"] == sis1_i_n) & (relation_df["object_number2"] == sis1_o_n)).any():
            rel_parent_i_n = np.asscalar(
                relation_df.loc[
                    (relation_df["image_number2"] == sis1_i_n) & (relation_df["object_number2"] == sis1_o_n),
                    "image_number1",
                ].values
            )
            rel_parent_o_n = np.asscalar(
                relation_df.loc[
                    (relation_df["image_number2"] == sis1_i_n) & (relation_df["object_number2"] == sis1_o_n),
                    "object_number1",
                ].values
            )
            rel_pairs.append([rel_parent_i_n, rel_parent_o_n, sis1_i_n, sis1_o_n])
            if [rel_parent_i_n, rel_parent_o_n, sis1_i_n, sis1_o_n] in false_link:
                mis_link = 1
        else:
            mis_link = 1

        if ((relation_df["image_number2"] == sis2_i_n) & (relation_df["object_number2"] == sis2_o_n)).any():
            rel_parent_i_n = np.asscalar(
                relation_df.loc[
                    (relation_df["image_number2"] == sis2_i_n) & (relation_df["object_number2"] == sis2_o_n),
                    "image_number1",
                ].values
            )
            rel_parent_o_n = np.asscalar(
                relation_df.loc[
                    (relation_df["image_number2"] == sis2_i_n) & (relation_df["object_number2"] == sis2_o_n),
                    "object_number1",
                ].values
            )
            rel_pairs.append([rel_parent_i_n, rel_parent_o_n, sis2_i_n, sis2_o_n])
            if [rel_parent_i_n, rel_parent_o_n, sis2_i_n, sis2_o_n] in false_link:
                mis_link = 1
        else:
            mis_link = 1

        if mis_link == 0:  # means both candi_am_sisters have their own mother
            nm_ind.append(i)
        else:
            mitosis_pairs_to_break.extend(rel_pairs)
    for i in sorted(nm_ind, reverse=True):
        del candi_am_sisters[i]
    return candi_am_sisters, mitosis_pairs_to_break


# --------remove non-fuse pairs-------
# if each fuse parent has rel-child in Per-Relationships and not in false
# link or border obj, they are right link


def find_fuse_pairs_to_break(relation_df, postfuse_cells, prefuse_group, false_link, border_obj):
    nf_ind = []
    fuse_pairs_to_break = []
    fuse_pairs = []
    for i in range(len(postfuse_cells)):
        mis_link = 0
        f_pairs = []
        rel_pairs = []
        border_count = 0

        f_group = prefuse_group[i]
        fc_i_n = postfuse_cells[i][0]
        fc_o_n = postfuse_cells[i][1]
        if [fc_i_n, fc_o_n] in border_obj:
            border_count += 1
        if ((relation_df["image_number2"] == fc_i_n) & (relation_df["object_number2"] == fc_o_n)).any():
            rel_parent_i_n = np.asscalar(
                relation_df.loc[
                    (relation_df["image_number2"] == fc_i_n) & (relation_df["object_number2"] == fc_o_n),
                    "image_number1",
                ].values
            )
            rel_parent_o_n = np.asscalar(
                relation_df.loc[
                    (relation_df["image_number2"] == fc_i_n) & (relation_df["object_number2"] == fc_o_n),
                    "object_number1",
                ].values
            )
            rel_pairs.append([rel_parent_i_n, rel_parent_o_n, fc_i_n, fc_o_n])
            if [rel_parent_i_n, rel_parent_o_n, fc_i_n, fc_o_n] in false_link:
                mis_link = 1
        else:
            mis_link = 1

        for j in range(len(f_group)):
            fp_i_n = f_group[j][0]
            fp_o_n = f_group[j][1]
            if [fp_i_n, fp_o_n] in border_obj:
                border_count += 1

            f_pairs.append([fp_i_n, fp_o_n, fc_i_n, fc_o_n])
            if ((relation_df["image_number1"] == fp_i_n) & (relation_df["object_number1"] == fp_o_n)).any():
                rel_child_i_n = np.asscalar(
                    relation_df.loc[
                        (relation_df["image_number1"] == fp_i_n) & (relation_df["object_number1"] == fp_o_n),
                        "image_number2",
                    ].values
                )
                rel_child_o_n = np.asscalar(
                    relation_df.loc[
                        (relation_df["image_number1"] == fp_i_n) & (relation_df["object_number1"] == fp_o_n),
                        "object_number2",
                    ].values
                )
                rel_pairs.append([fp_i_n, fp_o_n, rel_child_i_n, rel_child_o_n])
                if [fp_i_n, fp_o_n, rel_child_i_n, rel_child_o_n] in false_link:
                    mis_link = 1
            else:
                mis_link = 1
        if mis_link == 0 or border_count >= 2:
            nf_ind.append(i)
        else:
            fuse_pairs_to_break.extend(rel_pairs)
            fuse_pairs.extend(f_pairs)
    if len(fuse_pairs_to_break) > 0:
        fuse_pairs_to_break = np.unique(np.asarray(fuse_pairs_to_break), axis=0).tolist()
    for i in sorted(nf_ind, reverse=True):
        del postfuse_cells[i]
        del prefuse_group[i]
    return postfuse_cells, prefuse_group, fuse_pairs, fuse_pairs_to_break


# remove non-split pairs
def find_split_pairs_to_break(relation_df, presplit_cells, postsplit_group, false_link, border_obj):
    ns_ind = []
    split_pairs_to_break = []
    split_pairs = []
    for i in range(len(presplit_cells)):
        mis_link = 0
        s_pairs = []
        rel_pairs = []
        border_count = 0

        s_group = postsplit_group[i]
        sp_i_n = presplit_cells[i][0]
        sp_o_n = presplit_cells[i][1]
        if [sp_i_n, sp_o_n] in border_obj:
            border_count += 1

        if ((relation_df["image_number1"] == sp_i_n) & (relation_df["object_number1"] == sp_o_n)).any():
            rel_child_i_n = np.asscalar(
                relation_df.loc[
                    (relation_df["image_number1"] == sp_i_n) & (relation_df["object_number1"] == sp_o_n),
                    "image_number2",
                ].values
            )
            rel_child_o_n = np.asscalar(
                relation_df.loc[
                    (relation_df["image_number1"] == sp_i_n) & (relation_df["object_number1"] == sp_o_n),
                    "object_number2",
                ].values
            )
            rel_pairs.append([sp_i_n, sp_o_n, rel_child_i_n, rel_child_o_n])
            if [sp_i_n, sp_o_n, rel_child_i_n, rel_child_o_n] in false_link:
                mis_link = 1
        else:
            mis_link = 1

        for j in range(len(s_group)):
            sc_i_n = s_group[j][0]
            sc_o_n = s_group[j][1]
            if [sc_i_n, sc_o_n] in border_obj:
                border_count += 1
            s_pairs.append([sp_i_n, sp_o_n, sc_i_n, sc_o_n])
            if ((relation_df["image_number2"] == sc_i_n) & (relation_df["object_number2"] == sc_o_n)).any():
                rel_parent_i_n = np.asscalar(
                    relation_df.loc[
                        (relation_df["image_number2"] == sc_i_n) & (relation_df["object_number2"] == sc_o_n),
                        "image_number1",
                    ].values
                )
                rel_parent_o_n = np.asscalar(
                    relation_df.loc[
                        (relation_df["image_number2"] == sc_i_n) & (relation_df["object_number2"] == sc_o_n),
                        "object_number1",
                    ].values
                )
                rel_pairs.append([rel_parent_i_n, rel_parent_o_n, sc_i_n, sc_o_n])
                if [rel_parent_i_n, rel_parent_o_n, sc_i_n, sc_o_n] in false_link:
                    mis_link = 1
            else:
                mis_link = 1
        if mis_link == 0 or border_count >= 2:
            ns_ind.append(i)
        else:
            split_pairs_to_break.extend(rel_pairs)
            split_pairs.extend(s_pairs)
    if len(split_pairs_to_break) > 0:
        split_pairs_to_break = np.unique(np.asarray(split_pairs_to_break), axis=0).tolist()
    for i in sorted(ns_ind, reverse=True):
        del presplit_cells[i]
        del postsplit_group[i]
    return presplit_cells, postsplit_group, split_pairs, split_pairs_to_break


# -------------judge fuse type-----------
# two fuse types:
# 1 undersegmentation:two or more cells fuse together:
# 2 oversegmentation:
# (1)several fragment of a cell join together
# (2) one single cell and one fragment


def judge_fuse_type(df, am_record, fc_cell, fp_group, fc_prob, fp_group_prob, tracklet_len_thres=5):
    false_label = []
    mitosis_fc_label = []
    mitosis_fp_label = []
    mitosis_fp_group = []
    mitosis_fp_group_xy = []
    mitosis_fuse_flag = 0

    type_list = ["single", "fragment", "multi"]

    fc_i_n, fc_o_n = fc_cell[0], fc_cell[1]
    fc_sure = 0

    fc_label = np.asscalar(
        df.loc[(df["ImageNumber"] == fc_i_n) & (df["ObjectNumber"] == fc_o_n), "Cell_TrackObjects_Label"].values
    )
    fc_arr = df.loc[df["Cell_TrackObjects_Label"] == fc_label, "Cell_AreaShape_Area"].values
    fc_traj_len = len(fc_arr)
    fc_am_flag = judge_traj_am(df, am_record, fc_i_n, fc_o_n, judge_later=True, t_range=3)

    fc_prob[1] = 0  # impossible to be a fragment
    if fc_traj_len > tracklet_len_thres:
        fc_prob[2] = 0  # highly unlikely to be a multi_cells
        fc_sure = 1

    fc_type = type_list[np.argmax(fc_prob)]

    nb_fp = len(fp_group)
    new_fp_group_prob = []
    fp_group_label = []
    fp_group_type = []
    fp_group_sure = []
    fp_group_traj_len = []
    fp_group_am_flag = []
    fp_group_size = []
    # fp_group_ff=[]
    fp_group_xy = []
    size_simi = 0

    for [fp_i_n, fp_o_n], fp_prob in zip(fp_group, fp_group_prob):
        fp_label = np.asscalar(
            df.loc[(df["ImageNumber"] == fp_i_n) & (df["ObjectNumber"] == fp_o_n), "Cell_TrackObjects_Label"].values
        )
        fp_arr = df.loc[df["Cell_TrackObjects_Label"] == fp_label, "Cell_AreaShape_Area"].values
        fp_size = np.asscalar(
            df.loc[(df["ImageNumber"] == fp_i_n) & (df["ObjectNumber"] == fp_o_n), "Cell_AreaShape_Area"].values
        )
        # fp_ff=np.asscalar(df.loc[(df['ImageNumber']==fp_i_n)&(df['ObjectNumber']==fp_o_n)]['Cell_AreaShape_FormFactor'].values)

        fp_xy = (
            df.loc[
                (df["ImageNumber"] == fp_i_n) & (df["ObjectNumber"] == fp_o_n),
                ["Cell_AreaShape_Center_X", "Cell_AreaShape_Center_Y"],
            ]
            .values[0]
            .tolist()
        )
        fp_traj_len = len(fp_arr)
        # fp_size=np.asscalar(df.loc[(df['ImageNumber']==fp_i_n)&(df['ObjectNumber']==fp_o_n),'Cell_AreaShape_Area'].values)
        fp_am_flag = judge_traj_am(df, am_record, fp_i_n, fp_o_n, judge_later=False, t_range=3)
        fp_sure = 0

        fp_prob[2] = 0  # impossible to be a multi_cell

        if fp_traj_len > tracklet_len_thres:
            fp_prob = [1, 0, 0]
            fp_sure = 1
        if fp_am_flag == 2:
            fp_prob = [1, 0, 0]
            fp_sure = 1
        if fp_am_flag == 1:
            fp_prob = [1, 0, 0]
            fp_sure = 1

        fp_group_label.append(fp_label)
        fp_group_type.append(type_list[np.argmax(np.array(fp_prob))])
        fp_group_sure.append(fp_sure)
        fp_group_traj_len.append(fp_traj_len)
        fp_group_am_flag.append(fp_am_flag)
        fp_group_size.append(fp_size)
        # fp_group_ff.append(fp_ff)
        fp_group_xy.append(fp_xy)

        new_fp_group_prob.append(fp_prob)

    fp_group_prob = new_fp_group_prob

    if nb_fp == 2:
        size_simi = 1 - abs(fp_group_size[0] - fp_group_size[1]) * 1.0 / (fp_group_size[0] + fp_group_size[1])
        if sum(fp_group_am_flag) > 1 and size_simi > 0.8:
            mitosis_fuse_flag = 1

    if mitosis_fuse_flag == 1:
        mitosis_fc_label.append(fc_label)
        mitosis_fp_label = fp_group_label
        mitosis_fp_group = fp_group
        mitosis_fp_group_xy = fp_group_xy

    if fc_type == "single" and fp_group_type.count("single") <= 1:
        for i in range(nb_fp):
            if fp_group_type[i] == "fragment":
                false_label.append(fp_group_label[i])

    if fc_type == "single" and fp_group_type.count("single") > 1:
        if fc_sure == 1 and fp_group_sure.count(1) < 1:
            for i in range(nb_fp):
                false_label.append(fp_group_label[i])

        if fc_sure == 1 and fp_group_sure.count(1) == 1:
            idx = fp_group_sure.index(1)
            for i in range(nb_fp):
                if i != idx:
                    false_label.append(fp_group_label[i])
            #         fp_group_type[idx]='fragment'
            # for i in range(nb_fp):
            #     if fp_group_type[i]=='fragment':
            #         false_label.append(fp_group_label[i])

        if fc_sure == 1 and fp_group_sure.count(1) > 1:
            if fc_traj_len < max(fp_group_traj_len):
                false_label.append(fc_label)
            else:
                for i in range(nb_fp):
                    false_label.append(fp_group_label[i])

        if fc_sure == 0 and fp_group_sure.count(1) < 1:
            false_label.append(fc_label)
            for i in range(nb_fp):
                false_label.append(fp_group_label[i])

        if fc_sure == 0 and fp_group_sure.count(1) == 1:
            false_label.append(fc_label)
            idx = fp_group_sure.index(1)
            for i in range(nb_fp):
                if i != idx:
                    false_label.append(fp_group_label[i])
            #         fp_group_type[idx]='fragment'
            # for i in range(nb_fp):
            #     if fp_group_type[i]=='fragment':
            #         false_label.append(fp_group_label[i])

        if fc_sure == 0 and fp_group_sure.count(1) > 1:
            false_label.append(fc_label)
            for i in range(nb_fp):
                if fp_group_type[i] == "fragment":
                    false_label.append(fp_group_label[i])

    if fc_type == "multi" and fp_group_type.count("single") > 1:
        false_label.append(fc_label)
        for i in range(nb_fp):
            if fp_group_type[i] == "fragment":
                false_label.append(fp_group_label[i])

    if fc_type == "multi" and fp_group_type.count("single") <= 1:
        print("uncertain")
        if fp_group_sure.count(1) == 1:
            false_label.append(fc_label)
            idx = fp_group_sure.index(1)
            for i in range(nb_fp):
                if i != idx:
                    false_label.append(fp_group_label[i])
        if fp_group_sure.count(1) == 0:
            false_label.append(fc_label)
            for i in range(nb_fp):
                false_label.append(fp_group_label[i])

    return (
        false_label,
        mitosis_fc_label,
        mitosis_fp_label,
        mitosis_fp_group,
        mitosis_fp_group_xy,
        fc_type,
        fp_group_type,
    )


# -------------judge split type-----------
# 4 split types:
# 1 undersegmentation before split: two or more cells fuse and split
# 2 oversegmentation of a cell:
# (1)two fragments
# (2)one fragment with one cell(large cell)
# 3 cell mitosis


def judge_split_type(df, am_record, sp_cell, sc_group, sp_prob, sc_group_prob, tracklet_len_thres=5):
    false_label = []
    candi_mitosis_label = []
    false_mitosis_obj = []

    type_list = ["single", "fragment", "multi"]
    candi_mitosis_flag = 0

    sp_i_n, sp_o_n = sp_cell[0], sp_cell[1]
    sp_sure = 0

    sp_label = np.asscalar(
        df.loc[(df["ImageNumber"] == sp_i_n) & (df["ObjectNumber"] == sp_o_n), "Cell_TrackObjects_Label"].values
    )
    sp_arr = df.loc[df["Cell_TrackObjects_Label"] == sp_label, "Cell_AreaShape_Area"].values
    sp_traj_len = len(sp_arr)
    sp_am_flag = judge_traj_am(df, am_record, sp_i_n, sp_o_n, judge_later=False, t_range=3)

    sp_prob[1] = 0  # impossible to be a fragment
    if sp_traj_len > tracklet_len_thres:
        sp_prob[2] = 0
        sp_sure = 1

    sp_type = type_list[np.argmax(sp_prob)]

    nb_sc = len(sc_group)

    new_sc_group_prob = []
    sc_group_label = []
    sc_group_type = []
    sc_group_sure = []

    sc_group_am_flag = []
    sc_group_size = []
    sc_group_traj_len = []  # record the sc whose traj_len>tracklet_len_thres
    size_simi = 0

    for [sc_i_n, sc_o_n], sc_prob in zip(sc_group, sc_group_prob):
        sc_label = np.asscalar(
            df.loc[(df["ImageNumber"] == sc_i_n) & (df["ObjectNumber"] == sc_o_n), "Cell_TrackObjects_Label"].values
        )
        sc_arr = df.loc[df["Cell_TrackObjects_Label"] == sc_label, "Cell_AreaShape_Area"].values
        sc_traj_len = len(sc_arr)
        sc_size = np.asscalar(
            df.loc[(df["ImageNumber"] == sc_i_n) & (df["ObjectNumber"] == sc_o_n), "Cell_AreaShape_Area"].values
        )

        sc_am_flag = judge_traj_am(df, am_record, sc_i_n, sc_o_n, judge_later=True, t_range=3)
        sc_sure = 0

        sc_prob[2] = 0  # impossible to be a multi_cell
        if sc_traj_len > tracklet_len_thres:
            sc_prob = [1, 0, 0]
            sc_sure = 1
        if sc_am_flag == 2:
            sc_prob = [1, 0, 0]
            sc_sure = 1
        if sc_am_flag == 1:
            sc_prob = [1, 0, 0]
            sc_sure = 1
        sc_group_am_flag.append(sc_am_flag)
        sc_group_size.append(sc_size)

        sc_group_label.append(sc_label)
        sc_group_type.append(type_list[np.argmax(np.array(sc_prob))])
        sc_group_sure.append(sc_sure)

        new_sc_group_prob.append(sc_prob)

        if sc_traj_len > tracklet_len_thres:
            sc_group_traj_len.append(1)
        else:
            sc_group_traj_len.append(0)

    sc_group_prob = new_sc_group_prob
    # if sp_am_flag==2 and ((sc_group_am_flag.count(0)==1 and size_simi>0.8)
    # or (sc_group_am_flag.count(0)==0 and nb_sc==2)):
    if (
        sp_am_flag == 2
        or sum(sc_group_am_flag) > 0
        or (sp_traj_len > tracklet_len_thres and sum(sc_group_traj_len) == 2)
    ):
        candi_mitosis_flag = 1

    if candi_mitosis_flag == 0:
        # if nb_sc>=3 or (sp_traj_len<=tracklet_len_thres and sum(sc_group_traj_len)>=2): undersegmentation
        if sp_type == "single" and sc_group_type.count("single") <= 1:
            for i in range(nb_sc):
                if sc_group_type[i] == "fragment":
                    false_label.append(sc_group_label[i])

        if sp_type == "single" and sc_group_type.count("single") > 1:

            if sp_sure == 1 and sc_group_sure.count(1) < 1:
                for i in range(nb_sc):
                    false_label.append(sc_group_label[i])

            if sp_sure == 1 and sc_group_sure.count(1) == 1:
                idx = sc_group_sure.index(1)
                for i in range(nb_sc):
                    if i != idx:
                        false_label.append(sc_group_label[i])
                #         sc_group_type[idx]='fragment'
                # for i in range(nb_sc):
                #     if sc_group_type[i]=='fragment':
                #         false_label.append(sc_group_label[i])

            if sp_sure == 1 and sc_group_sure.count(1) > 1:
                candi_mitosis_label.append(sp_label)
                for i in range(nb_sc):
                    candi_mitosis_label.append(sc_group_label[i])

            if sp_sure == 0 and sc_group_sure.count(1) < 1:
                false_label.append(sp_label)
                for i in range(nb_sc):
                    false_label.append(sc_group_label[i])

            if sp_sure == 0 and sc_group_sure.count(1) == 1:
                false_label.append(sp_label)
                idx = sc_group_sure.index(1)
                for i in range(nb_sc):
                    if i != idx:
                        false_label.append(sc_group_label[i])
                #         sc_group_type[idx]='fragment'
                # for i in range(nb_sc):
                #     if sc_group_type[i]=='fragment':
                #         false_label.append(sc_group_label[i])

            if sp_sure == 0 and sc_group_sure.count(1) > 1:
                false_label.append(sp_label)
                for i in range(nb_sc):
                    if sc_group_type[i] == "fragment":
                        false_label.append(sc_group_label[i])

        if sp_type == "multi" and sc_group_type.count("single") <= 1:
            print("uncertain")
            if sc_group_sure.count(1) == 1:
                false_label.append(sp_label)
                idx = sc_group_sure.index(1)
                for i in range(nb_sc):
                    if i != idx:
                        false_label.append(sc_group_label[i])
            if sc_group_sure.count(1) == 0:
                false_label.append(sp_label)
                for i in range(nb_sc):
                    false_label.append(sc_group_label[i])

        if sp_type == "multi" and sc_group_type.count("single") > 1:
            false_label.append(sp_label)
            for i in range(nb_sc):
                if sc_group_type[i] == "fragment":
                    false_label.append(sc_group_label[i])

    else:
        candi_mitosis_label.append(sp_label)
        for i in range(nb_sc):
            candi_mitosis_label.append(sc_group_label[i])

        if sp_type == "single":
            for i in range(nb_sc):
                if sc_group_type[i] == "fragment":
                    false_mitosis_obj.append(sc_group[i])

        # if sp_type=='single' and sc_group_type.count('single')<=1:
        #     for i in range(nb_sc):
        #         if sc_group_type[i]=='fragment':
        #             false_mitosis_obj.append(sc_group[i])

        # if sp_type=='single' and sc_group_type.count('single')>1:

        #     if sp_sure==1 and sc_group_sure.count(1)<1:
        #         for i in range(nb_sc):
        #             if sc_group_type[i]=='fragment':
        #                 false_mitosis_obj.append(sc_group[i])

        #     if sp_sure==1 and sc_group_sure.count(1)==1:
        #         for i in range(nb_sc):
        #             if sc_group_type[i]=='fragment':
        #                 false_mitosis_obj.append(sc_group[i])

        #     if sp_sure==1 and sc_group_sure.count(1)>1:
        #         print('mitosis')
        #         for i in range(nb_sc):
        #             if sc_group_type[i]=='fragment':
        #                 false_mitosis_obj.append(sc_group[i])

        #     if sp_sure==0 and sc_group_sure.count(1)<1:
        #         for i in range(nb_sc):
        #             if sc_group_type[i]=='fragment':
        #                 false_mitosis_obj.append(sc_group[i])

        #     if sp_sure==0 and sc_group_sure.count(1)==1:
        #         for i in range(nb_sc):
        #             if sc_group_type[i]=='fragment':
        #                 false_mitosis_obj.append(sc_group[i])

        #     if sp_sure==0 and sc_group_sure.count(1)>1:
        #         for i in range(nb_sc):
        #             if sc_group_type[i]=='fragment':
        #                 false_mitosis_obj.append(sc_group[i])

        if sp_type == "multi" and sc_group_type.count("single") <= 1:
            print("uncertain")
            if sc_group_sure.count(1) == 1:
                false_mitosis_obj.append(sp_cell)
                idx = sc_group_sure.index(1)
                for i in range(nb_sc):
                    if i != idx:
                        false_mitosis_obj.append(sc_group[i])
            if sc_group_sure.count(1) == 0:
                false_mitosis_obj.append(sp_cell)
                for i in range(nb_sc):
                    false_mitosis_obj.append(sc_group[i])

        if sp_type == "multi" and sc_group_type.count("single") > 1:
            false_mitosis_obj.append(sp_cell)
            for i in range(nb_sc):
                if sc_group_type[i] == "fragment":
                    false_mitosis_obj.append(sc_group[i])

    return candi_mitosis_flag, false_label, candi_mitosis_label, false_mitosis_obj, sp_type, sc_group_type


# find unique values and their indexes
def find_uni(inds_arr):
    idx_sort = np.argsort(inds_arr)
    sorted_inds_arr = inds_arr[idx_sort]
    vals, idx_start, count = np.unique(sorted_inds_arr, return_counts=True, return_index=True)

    # sets of indices
    res = np.split(idx_sort, idx_start[1:])
    # filter them with respect to their size, keeping only items occurring
    # more than once

    vals = vals[count > 0]

    res = list(filter(lambda x: x.size > 0, res))
    return vals, res


def get_mitotic_triple_scores(F, L, mitosis_max_distance=75, size_simi_thres=0.7):
    """Compute scores for matching a parent to two daughters

    F - an N x 3 (or more) array giving X, Y and frame # of the first object

        in each track

    L - an N x 3 (or more) array giving X, Y and frame # of the last object

        in each track

    Returns: an M x 3 array of M triples where the first column is the

             index in the L array of the parent cell and the remaining

             columns are the indices of the daughters in the F array


             an M-element vector of distances of the parent from the expected

    """

    X = 0

    Y = 1

    IIDX = 2  # img_num

    OIDX = 3  # obj_num

    AIDX = 4  # area

    if len(F) <= 1:

        return np.array([]), np.array([]), np.array([])

    max_distance = mitosis_max_distance

    # Find all daughter pairs within same frame

    i, j = np.where(F[:, np.newaxis, IIDX] == F[np.newaxis, :, IIDX])

    i, j = i[i < j], j[i < j]  # get rid of duplicates and self-compares

    # Calculate the maximum allowed distance before one or the other

    # daughter is farther away than the maximum allowed from the center

    #

    # That's the max_distance * 2 minus the distance

    #

    dmax = max_distance * 2 - np.sqrt(np.sum((F[i, :2] - F[j, :2]) ** 2, 1))

    dist_mask = dmax >= 0

    i, j, dmax = i[dist_mask], j[dist_mask], dmax[dist_mask]

    # use size similarity to exclude candidate pairs
    size_simi = 1 - abs((F[i, AIDX] - F[j, AIDX]) * 1.0 / (F[i, AIDX] + F[j, AIDX]))
    size_mask = np.zeros(i.shape, dtype=bool)

    uni_t = np.unique(F[i, IIDX])
    ii = np.arange(len(i))
    for ti in uni_t:
        match_inds = []
        mask_ti = F[i[ii], IIDX] == ti
        # print(ii[mask_ti])
        ti_size_simi = size_simi[ii[mask_ti]]
        # print(ti_size_simi)
        sort_inds = np.argsort(-ti_size_simi)  # from large to small
        # print(ti_size_simi[sort_inds])
        for si in sort_inds:
            # if this size_simi<thres, no pairs after meet the requirement
            if ti_size_simi[si] < size_simi_thres:
                break
            ii_where = np.where(size_simi == ti_size_simi[si])[0]
            # print(ii_where)
            i_where = i[ii_where][0]

            j_where = j[ii_where][0]
            # print(i_where,j_where,match_inds)
            # if in match_inds means this one already have a sister with larger
            # size_simi
            if i_where in match_inds or j_where in match_inds:
                continue
            else:
                match_inds.append(i_where)
                match_inds.append(j_where)
                size_mask[ii_where] = True

    # the origin function in cellprofiler.trackobjects doesn't have
    # dmax=dmax[mask]
    i, j, dmax = i[size_mask], j[size_mask], dmax[size_mask]
    size_simi = size_simi[size_mask]

    if len(i) == 0:

        return np.array([]), np.array([]), np.array([])

    center_x = (F[i, X] + F[j, X]) / 2

    center_y = (F[i, Y] + F[j, Y]) / 2

    # frame = F[i, IIDX]

    # Find all parent-daughter pairs where the parent

    # is in the frame previous to the daughters

    ij, k = [_.flatten() for _ in np.mgrid[0 : len(i), 0 : len(L)]]

    mask = F[i[ij], IIDX] == L[k, IIDX] + 1

    ij, k = ij[mask], k[mask]

    if len(ij) == 0:

        return np.array([]), np.array([]), np.array([])

    d = np.sqrt((center_x[ij] - L[k, X]) ** 2 + (center_y[ij] - L[k, Y]) ** 2)
    # find the group with smallest d
    vals, res = find_uni(ij)
    mask = np.zeros(ij.shape, dtype=bool)
    if len(vals) > 0:
        for vi in range(len(vals)):
            rep_v = vals[vi]
            rep_inds = res[vi]

            min_ind = rep_inds[np.argmin(d[rep_inds])]
            if d[min_ind] < dmax[ij[min_ind]]:
                mask[min_ind] = True

    # mask = d <= dmax[ij]

    ij, k, d = ij[mask], k[mask], d[mask]

    if len(ij) == 0:

        return np.array([]), np.array([]), np.array([])

    #         rho = calculate_area_penalty(

    #             F[i[ij], AIDX] + F[j[ij], AIDX], L[k, AIDX])

    # return np.column_stack((i[ij], j[ij], k)), d * rho
    return np.column_stack((L[k, 2:4], F[i[ij], 2:4], F[j[ij], 2:4])), d, size_simi[ij]


def search_wrong_mitosis(mitosis_record, mature_time):

    mother_label = mitosis_record["mother_traj_label"].values
    daughter_label = mitosis_record[["sis1_traj_label", "sis2_traj_label"]].values.flatten()
    wrong_mitosis_inds = []
    # find mother label that have multiple paris of daughters
    uni_m_label, counts = np.unique(mother_label, return_index=False, return_inverse=False, return_counts=True)
    rep_m_label = uni_m_label[counts > 1]
    for rm_lable in rep_m_label:
        inds = mitosis_record.loc[mitosis_record["mother_traj_label"] == rm_lable].index.tolist()
        wrong_mitosis_inds.extend(inds)
    # find cell that have just been born and divide again
    for d_label in daughter_label:
        if d_label in mother_label.tolist():
            birth_time = mitosis_record.loc[mitosis_record["sis1_traj_label"] == d_label, "sis1_i_n"].values
            if len(birth_time) == 0:
                birth_time = mitosis_record.loc[mitosis_record["sis2_traj_label"] == d_label, "sis2_i_n"].values

            match_list = mitosis_record[mitosis_record["mother_traj_label"] == d_label].index.tolist()

            for ind in match_list:
                give_birth_time = mitosis_record.loc[ind, "mother_i_n"]
                give_birth_obj = mitosis_record.loc[ind, "mother_o_n"]
                if (give_birth_time - birth_time) < mature_time:
                    wrong_mitosis_inds.append(ind)
    mitosis_record = mitosis_record.drop(np.unique(np.array(wrong_mitosis_inds)))
    return mitosis_record


# return numpy record of each traj start and traj end. [img_num obj_num]
def traj_start_end_info(df):
    traj_labels = df["Cell_TrackObjects_Label"].values
    traj_labels = np.sort(np.unique(traj_labels[traj_labels > 0]))
    traj_quan = len(traj_labels)  # the quantity of trajectories
    print("traj quantity", traj_quan)

    traj_start = []
    traj_start_xy = []
    traj_start_area = []
    traj_end = []
    traj_end_xy = []
    traj_end_area = []
    for cur_traj_label in traj_labels:
        traj_start.append(
            df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, ["ImageNumber", "ObjectNumber"]].values[0].tolist()
        )
        traj_end.append(
            df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, ["ImageNumber", "ObjectNumber"]].values[-1].tolist()
        )
        traj_start_xy.append(
            df.loc[
                df["Cell_TrackObjects_Label"] == cur_traj_label, ["Cell_AreaShape_Center_X", "Cell_AreaShape_Center_Y"]
            ]
            .values[0]
            .tolist()
        )
        traj_end_xy.append(
            df.loc[
                df["Cell_TrackObjects_Label"] == cur_traj_label, ["Cell_AreaShape_Center_X", "Cell_AreaShape_Center_Y"]
            ]
            .values[-1]
            .tolist()
        )
        traj_start_area.append(
            df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, "Cell_AreaShape_Area"].values[0].tolist()
        )
        traj_end_area.append(
            df.loc[df["Cell_TrackObjects_Label"] == cur_traj_label, "Cell_AreaShape_Area"].values[-1].tolist()
        )
    traj_start = np.array(traj_start)
    traj_end = np.array(traj_end)
    traj_start_xy = np.array(traj_start_xy)
    traj_end_xy = np.array(traj_end_xy)
    traj_start_area = np.array(traj_start_area)
    traj_end_area = np.array(traj_end_area)
    return traj_start, traj_end, traj_start_xy, traj_end_xy, traj_start_area, traj_end_area


def calculate_area_penalty(a1, a2):
    """Calculate a penalty for areas that don't match
    Ideally, area should be conserved while tracking. We divide the larger
    of the two by the smaller of the two to get the area penalty
    which is then multiplied by the distance.
    Note that this differs from Jaqaman eqn 5 which has an asymmetric
    penalty (sqrt((a1 + a2) / b) for a1+a2 > b and b / (a1 + a2) for
    a1+a2 < b. I can't think of a good reason why they should be
    asymmetric.
    """
    result = a1 / a2
    result[result < 1] = 1 / result[result < 1]
    result[np.isnan(result)] = np.inf
    return result


def get_gap_pair_scores(F, L, max_gap):
    """Compute scores for matching last frame with first to close gaps
    F - an N x 3 (or more) array giving X, Y and frame # of the first object
        in each track
    L - an N x 3 (or more) array giving X, Y and frame # of the last object
        in each track
    max_gap - the maximum allowed # of frames between the last and first
    Returns: an M x 2 array of M pairs where the first element of the array
             is the index of the track whose last frame is to be joined to
             the track whose index is the second element of the array.
             an M-element vector of scores.
    """
    #
    # There have to be at least two things to match
    #

    if F.shape[0] <= 1:
        return np.array([]), np.array([])

    X = 0
    Y = 1
    IIDX = 2
    AIDX = 4

    #
    # Create an indexing ordered by the last frame index and by the first
    #
    i = np.arange(len(F))
    j = np.arange(len(F))
    f_iidx = F[:, IIDX].astype(int)
    l_iidx = L[:, IIDX].astype(int)

    i_lorder = np.lexsort((i, l_iidx))
    j_forder = np.lexsort((j, f_iidx))
    i = i[i_lorder]
    j = j[j_forder]
    i_counts = np.bincount(l_iidx)
    j_counts = np.bincount(f_iidx)
    i_indexes = Indexes([i_counts])
    j_indexes = Indexes([j_counts])
    #
    # The lowest possible F for each L is 1+L
    #
    j_self = np.minimum(np.arange(len(i_counts)), len(j_counts) - 1)
    j_first_idx = j_indexes.fwd_idx[j_self] + j_counts[j_self]
    #
    # The highest possible F for each L is L + max_gap. j_end is the
    # first illegal value... just past that.
    #
    j_last = np.minimum(np.arange(len(i_counts)) + max_gap, len(j_counts) - 1)
    j_end_idx = j_indexes.fwd_idx[j_last] + j_counts[j_last]
    #
    # Structure the i and j block ranges
    #
    ij_counts = j_end_idx - j_first_idx
    ij_indexes = Indexes([i_counts, ij_counts])
    if ij_indexes.length == 0:
        return np.array([]), np.array([])
    #
    # The index into L of the first element of the pair
    #
    ai = i[i_indexes.fwd_idx[ij_indexes.rev_idx] + ij_indexes.idx[0]]
    #
    # The index into F of the second element of the pair
    #
    aj = j[j_first_idx[ij_indexes.rev_idx] + ij_indexes.idx[1]]
    #
    # The distances
    #
    d = np.sqrt((L[ai, X] - F[aj, X]) ** 2 + (L[ai, Y] - F[aj, Y]) ** 2)
    #
    # Rho... the area penalty
    #
    rho = calculate_area_penalty(L[ai, AIDX], F[aj, AIDX])
    return np.column_stack((ai, aj)), d * rho


# ---high_traj_len_thres and low_traj_len_thres depend on the expriment time interval


def cal_size_correlation(df, img_num, sis1_o_n, sis2_o_n, high_traj_len_thres=4, low_traj_len_thres=3):
    sis1_traj_label = np.asscalar(
        df.loc[(df["ImageNumber"] == img_num) & (df["ObjectNumber"] == sis1_o_n)]["Cell_TrackObjects_Label"].values
    )
    sis1_size_arr = df.loc[df["Cell_TrackObjects_Label"] == sis1_traj_label, "Cell_AreaShape_Area"].values
    sis1_traj_len = len(sis1_size_arr)

    sis2_traj_label = np.asscalar(
        df.loc[(df["ImageNumber"] == img_num) & (df["ObjectNumber"] == sis2_o_n)]["Cell_TrackObjects_Label"].values
    )
    sis2_size_arr = df.loc[df["Cell_TrackObjects_Label"] == sis2_traj_label, "Cell_AreaShape_Area"].values
    sis2_traj_len = len(sis2_size_arr)

    if sis1_traj_len > high_traj_len_thres and sis2_traj_len > high_traj_len_thres:
        size_corr = pearsonr(sis1_size_arr[:high_traj_len_thres], sis2_size_arr[:high_traj_len_thres])[0]

    else:
        if sis1_traj_len < low_traj_len_thres or sis2_traj_len < low_traj_len_thres:
            size_corr = 0
        else:
            shorter_len = min(sis1_traj_len, sis2_traj_len)
            # print(sis1_size_arr,sis2_size_arr)
            size_corr = pearsonr(sis1_size_arr[:shorter_len], sis2_size_arr[:shorter_len])[0]
    return size_corr


# #this dimension order of skimage is the same as numpy array
# def check_around(df,img_path,img_list,img_num,obj_num,search_radius=250,back_t=3,forward_t=3):
#     neighbor_obj=[]
#     if np.asscalar(df.loc[(df['ImageNumber']==img_num)&(df['ObjectNumber']==obj_num),'Cell_TrackObjects_Label'].values)==-1:
#         return neighbor_obj

#     t_span=max(df['ImageNumber'])
#     img=imread(img_path+'/'+img_list[img_num-1])
#     rps=regionprops(img)
#     r_labels=[r.label for r in rps]
#     obj_i=r_labels.index(obj_num)
#     r_area=[r.area for r in rps]
#     r_centroid=[r.centroid for r in rps]
#     r_perimeter=[r.perimeter for r in rps]
#     obj_center=r_centroid[obj_i]
#     obj_area=r_area[obj_i]
#     obj_perimeter=r_perimeter[obj_i]
#     #obj_ff=4*pi*obj_area/(r_perimeter)**2#form factor

#     x_min=int(max(obj_center[0]-search_radius,0))
#     x_max=int(min(obj_center[0]+search_radius,img.shape[0]))
#     y_min=int(max(obj_center[1]-search_radius,0))
#     y_max=int(min(obj_center[1]+search_radius,img.shape[1]))


#     for look_i_n in range(max(img_num-back_t,1),min(img_num+forward_t,t_span)):#look_i_n:look image number
#         look_img=imread(img_path+'/'+img_list[look_i_n-1])
#         look_img_roi=look_img[x_min:x_max,y_min:y_max]
#         neighbor_obj_num=np.unique(look_img_roi).tolist()
#         neighbor_obj_num.remove(0)
#         for look_o_n in neighbor_obj_num:
#             if np.asscalar(df.loc[(df['ImageNumber']==look_i_n)&(df['ObjectNumber']==look_o_n),'Cell_TrackObjects_Label'].values)==-1:
#                 continue
#             if look_i_n!=img_num or look_o_n!=obj_num:
#                 neighbor_obj.append([look_i_n,look_o_n])
#     neighor_obj=np.array(neighbor_obj)
#     return neighbor_obj


# def candi_family(df,img_path,img_list,img_num,obj_num,traj_end,traj_start):
#     candi_family_pair=[]
#     #find neighbor cells in pre frame that have overlap of this cell or traj_end
#     #search range should be large,because cell tends move a lot during mitosis
#     pre_nei=check_around(df,img_path,img_list,img_num,obj_num,search_radius=200,back_t=5,forward_t=0)

#     candi_te_parent=[]
#     for pre_i_n,pre_o_n in pre_nei:
#         te_nei = traj_end[(traj_end[:,0] == pre_i_n) & (traj_end[:,1] == pre_o_n)]#traj_end
#         if len(te_nei)>0:
#             candi_te_parent.append([pre_i_n,pre_o_n])
#     candi_te_parent=np.array(candi_te_parent)
#     #find neighbor cells that is brother of this cell
#     #could be start of traj or child of candidate parent,
#     #should have similar size with current cell or co_overlap with current cell on candiate parent
#     #search range should be small

#     bro_nei=check_around(df,img_path,img_list,img_num,obj_num,search_radius=100,back_t=4,forward_t=5)
#     candi_ts_bro=[]
#     for bro_i_n,bro_o_n in bro_nei:
#         ts_nei=traj_start[(traj_start[:,0]==bro_i_n)&(traj_start[:,1]==bro_o_n)]#traj_start
#         if len(ts_nei)>0:
#             candi_ts_bro.append([bro_i_n,bro_o_n])
#     candi_ts_bro=np.array(candi_ts_bro)

#     for i in range(candi_te_parent.shape[0]):
#         for j in range(candi_ts_bro.shape[0]):
#             if candi_te_parent[i,0]<candi_ts_bro[j,0]:#pre_i_n should be smaller than bro_i_n
#                 candi_family_pair.append([candi_te_parent[i,0],candi_te_parent[i,1],candi_ts_bro[j,0],candi_ts_bro[j,1]])
#     return candi_family_pair


# def candi_trunk(df,img_path,img_list,img_num,obj_num):
#     # find traj that close to this obj,doesn't start or end in this time range

#     pre_i_n_arr=img_num-1
#     bro_i_n=img_num

#     candi_trunk_pair=[]
#     pre_traj_label_list=[]
#     pre_nei=check_around(df,img_path,img_list,img_num,obj_num,search_radius=100,back_t=3,forward_t=0)
#     for pre_i_n, pre_o_n in pre_nei:
#         pre_traj_label=np.asscalar(df.loc[(df['ImageNumber']==pre_i_n)&(df['ObjectNumber']==pre_o_n)]['Cell_TrackObjects_Label'].values)
#         pre_traj_start=df.loc[df['Cell_TrackObjects_Label']==pre_traj_label,'ImageNumber'].values[0]
#         #if img_num-pre_traj_start>3:
#         pre_traj_label_list.append(pre_traj_label)
#     pre_traj_label_list=list(set(pre_traj_label_list))
#     for trunk_label in pre_traj_label_list:
#         bro_o_n_arr=df.loc[(df['ImageNumber']==img_num)&(df['Cell_TrackObjects_Label']==trunk_label),'ObjectNumber'].values
#         pre_o_n_arr=np.array([])
#         if len(bro_o_n_arr)>0:
#             i=1
#             while (len(pre_o_n_arr)==0 and pre_i_n>0):
#                 pre_i_n=img_num-i
#                 pre_o_n_arr=df.loc[(df['ImageNumber']==pre_i_n)&(df['Cell_TrackObjects_Label']==trunk_label),'ObjectNumber'].values
#                 i+=1
#         if len(bro_o_n_arr)>0 and len(pre_o_n_arr)>0:
#             bro_o_n=np.asscalar(bro_o_n_arr)
#             pre_o_n=np.asscalar(pre_o_n_arr)
#             candi_trunk_pair.append([pre_i_n,pre_o_n,bro_i_n,bro_o_n])

#     return candi_trunk_pair


# #for search_underseg_link
# #1 search traj_end E and find its max_overlap child EC
# #2 find this max_overlap child's rel_parent ECP
# #3 if this ECP's max_overlap child ECPC is EC, this is a candidate underseg traj
# #4 search along this candi_underseg traj, if obj in this traj is max_overlap parent of a traj_start, it is an underseg traj
# def search_underseg_link(df,relation_df,am_record,img_path,img_list,traj_start,traj_end,false_link,us_t_range=3):
#     t_span=max(df['ImageNumber'])
#     pairs_need_break=[]
#     pairs_need_link=[]
#     us_link_pairs=[]#underseg_pair
#     underseg_obj=[]
#     candi_pairs=[]
#     for i in range(traj_end.shape[0]):
#         te_i_n=traj_end[i,0]#te:traj end
#         te_o_n=traj_end[i,1]
#         cur_traj_label=np.asscalar(df.loc[(df['ImageNumber']==te_i_n)&(df['ObjectNumber']==te_o_n)]['Cell_TrackObjects_Label'].values)
#         ot_length=len(df[df['Cell_TrackObjects_Label']==cur_traj_label].index.tolist())
#         obj_am_flag=judge_traj_am(df,am_record,te_i_n,te_o_n,judge_later=False)


#         if te_i_n==t_span:
#             continue
#         border_flag=judge_border(img_path,img_list,te_i_n,te_o_n)
#         if border_flag==1:
#             continue
#         if ot_length<5 and te_i_n>5 and obj_am_flag==0:#probably oversegmentation
#             continue
#         #us:under segmentation

#         us_notlink_start=[]#traj_end and underseg obj start
#         us_notlink_end=[]#underseg obj end and traj_start
#         us_link_start=[]
#         us_link_end=[]

#         te_ol_child=compute_overlap_single(img_path,img_list,te_i_n,te_o_n,te_i_n+1)
#         if np.all(te_ol_child==0):#if te obj have no overlap child
#             te_max_child_o_n=0
#         else:#te_max_child:traj end max ovelap child
#             te_max_child_o_n=np.argmax(te_ol_child)+1
#         #tmc:te_max_child
#         #rel_parent:parent in PerRelationships file
#         if te_max_child_o_n>0:
#             tmc_rel_parent=relation_df.loc[(relation_df['image_number2']==te_i_n+1)&(relation_df['object_number2']==te_max_child_o_n),'object_number1'].values
#             if len(tmc_rel_parent)>0:
#                 tmc_rel_parent=int(np.asscalar(tmc_rel_parent))
#                 #ol_child:overlap child
#                 tmc_rel_parent_ol_child=compute_overlap_single(img_path,img_list,te_i_n,tmc_rel_parent,te_i_n+1)


#                 #tmcrp:tmc_rel_parent
#                 if np.all(tmc_rel_parent_ol_child==0):#if traj end max ovelap child's rel parent have no overlap child
#                     tmcrp_max_child=0
#                 else:
#                     tmcrp_max_child=np.argmax(tmc_rel_parent_ol_child)+1
#                 #this traj end's max_overlap child and this child's rel parent's max overlap child is itself
#                 if tmcrp_max_child==te_max_child_o_n:#and te_ol_child[te_max_child_o_n-1]>0:???
#                     candi_underseg_label=np.asscalar(df.loc[(df['ImageNumber']==te_i_n+1)&(df['ObjectNumber']==te_max_child_o_n),'Cell_TrackObjects_Label'].values)
#                     candi_traj_length=len(df[df['Cell_TrackObjects_Label']==candi_underseg_label].index.tolist())
#                     if candi_traj_length<5 and te_i_n+1>5:#if the candi_traj_length is too short, it might not be an underseg
#                         continue


#                     candi_underseg_obj=df.loc[(df['ImageNumber']>=te_i_n+1)&(df['ImageNumber']<=te_i_n+us_t_range)&(df['Cell_TrackObjects_Label']==candi_underseg_label),['ImageNumber','ObjectNumber']].values.tolist()
#                     for candi_underseg_i_n, candi_underseg_o_n in candi_underseg_obj:
#                         ts_obj=traj_start[traj_start[:,0]==candi_underseg_i_n+1].tolist()
#                         for ts_i_n,ts_o_n in ts_obj:
#                             source_overlap=compute_overlap_single(img_path,img_list,ts_i_n,ts_o_n,candi_underseg_i_n)
#                             ts_max_parent_obj=np.argmax(source_overlap)+1
#                             if ts_max_parent_obj==candi_underseg_o_n:


#                                 us_notlink_start=[te_i_n,te_o_n,te_i_n+1,te_max_child_o_n]
#                                 us_notlink_end=[candi_underseg_i_n,candi_underseg_o_n,ts_i_n,ts_o_n]


#                                 rel_flag=-1
#                                 un_rel_flag=-1


#                                 us_link_start=[te_i_n,tmc_rel_parent,te_i_n+1,te_max_child_o_n]
#                                 #uso:underseg obj
#                                 uso_rel_child=relation_df.loc[(relation_df['image_number1']==candi_underseg_i_n)&(relation_df['object_number1']==candi_underseg_o_n),['image_number2','object_number2']].values.tolist()
#                                 #print(uso_rel_child)
#                                 if len(uso_rel_child)>0:
#                                     us_link_end=[candi_underseg_i_n,candi_underseg_o_n,uso_rel_child[0][0],uso_rel_child[0][1]]
#                                     #calculate the overlap of traj end and traj start
#                                     frame_overlap=compute_overlap_matrix(img_path,img_list,te_i_n,ts_i_n)
#                                     rel_flag=judge_mol_type(frame_overlap,tmc_rel_parent,uso_rel_child[0][1])
#                                     un_rel_flag=judge_mol_type(frame_overlap,te_o_n,ts_o_n)

#                                     pairs_need_break.append(us_link_start)
#                                     pairs_need_break.append(us_link_end)
#                                     us_link_pairs.append(us_link_start)
#                                     us_link_pairs.append(us_link_end)
#                                     us_link_pairs.append(us_notlink_start)
#                                     us_link_pairs.append(us_notlink_end)
#                                     underseg_obj.extend(df.loc[(df['ImageNumber']>=te_i_n+1)&(df['ImageNumber']<ts_i_n)&(df['Cell_TrackObjects_Label']==candi_underseg_label),['ImageNumber','ObjectNumber']].values.tolist())

# if rel_flag==3:#traj transfer during underseg

#                                         candi_pairs.append([te_i_n,te_o_n,uso_rel_child[0][0],uso_rel_child[0][1]])
#                                         candi_pairs.append([te_i_n,tmc_rel_parent,ts_i_n,ts_o_n])

#                                     else:#traj do not transfer, check the link is in false link or not
#                                         if us_link_start in false_link or us_link_end in false_link:
#                                             candi_pairs.append([te_i_n,te_o_n,uso_rel_child[0][0],uso_rel_child[0][1]])
#                                             candi_pairs.append([te_i_n,tmc_rel_parent,ts_i_n,ts_o_n])
#                                         else:
#                                             pairs_need_link.append([te_i_n,tmc_rel_parent,uso_rel_child[0][0],uso_rel_child[0][1]])
#                                             candi_pairs.append([te_i_n,te_o_n,ts_i_n,ts_o_n])


#                                 #     print('rel',rel_flag)
#                                 #     print(us_link_start)
#                                 #     print(us_link_end)

#                                 #     print('un_rel',un_rel_flag)
#                                 #     print(us_notlink_start)
#                                 #     print(us_notlink_end)

#                                 # print('==============================')
#                                 #break


#                         if len(us_notlink_start)>0:
#                             break
# return
# pairs_need_break,pairs_need_link,us_link_pairs,underseg_obj,candi_pairs
