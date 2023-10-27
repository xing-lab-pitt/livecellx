import numpy as np
from skimage.measure import label, regionprops


def generate_single_cell_img(img, seg, img_num, obj_num):
    # single_obj_mask=morphology.binary_dilation(seg==obj_num,morphology.disk(6))
    single_obj_mask = seg == obj_num
    single_obj_mask = label(single_obj_mask)
    rps = regionprops(single_obj_mask)
    candi_r = [r for r in rps if r.label == 1][0]
    candi_box = candi_r.bbox
    single_cell_img = single_obj_mask * img

    crop_cell_img = single_cell_img[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]
    crop_cell_img_env = img[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]

    crop_single_obj_mask = single_obj_mask[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]
    rps = regionprops(crop_single_obj_mask)
    candi_r = [r for r in rps if r.label == 1][0]
    center = candi_r.centroid

    return crop_cell_img, crop_cell_img_env


def find_mother(mitosis_df, traj_label):
    daughter_flag = 0
    if traj_label != -1:
        if (mitosis_df["sis1_traj_label"] == traj_label).any() or (mitosis_df["sis2_traj_label"] == traj_label).any():
            daughter_flag = 1
    return daughter_flag


def find_mother_label(mitosis_df, traj_label):
    if True in np.array(mitosis_df["sis1_traj_label"] == traj_label):
        ind = mitosis_df.loc[(mitosis_df["sis1_traj_label"] == traj_label)].index.tolist()[0]
        mother_label = int(mitosis_df["mother_traj_label"][ind])
    elif True in np.array(mitosis_df["sis2_traj_label"] == traj_label):
        ind = mitosis_df.loc[(mitosis_df["sis2_traj_label"] == traj_label)].index.tolist()[0]
        mother_label = int(mitosis_df["mother_traj_label"][ind])
    return mother_label


def find_offspring(df, mitosis_df, family_tree, traj_label):
    mother_label = traj_label
    if mother_label != -1 and (mitosis_df["mother_traj_label"] == mother_label).any():
        family_tree[int(mother_label)] = []
        sis1_label = mitosis_df.loc[mitosis_df["mother_traj_label"] == mother_label, "sis1_traj_label"].values[0]
        sis2_label = mitosis_df.loc[mitosis_df["mother_traj_label"] == mother_label, "sis2_traj_label"].values[0]
        if sis1_label != -1:
            family_tree[int(mother_label)].append(int(sis1_label))
        if sis2_label != -1:
            family_tree[int(mother_label)].append(int(sis2_label))
        family_tree = find_offspring(df, mitosis_df, family_tree, sis1_label)
        family_tree = find_offspring(df, mitosis_df, family_tree, sis2_label)
        return family_tree
    else:
        return family_tree


def parse(node, tree):
    if node not in tree:
        yield [node]
    else:
        for next_node in tree[node]:
            for r in parse(next_node, tree):
                yield [node] + r


def find_abnormal_fluor(traj_fluor, traj_t, peak_h=5):
    mask = traj_fluor != 0
    #     inds=np.where(traj_fluor!=0)[0]
    non0_traj_t = traj_t[mask]
    non0_traj_fluor = traj_fluor[mask]
    mean_fluct = np.mean(abs(np.diff(non0_traj_fluor)))

    ind1 = find_peaks(np.diff(non0_traj_fluor) / mean_fluct, height=peak_h)[0] + 1
    ind2 = (
        non0_traj_fluor.shape[0] - (find_peaks(np.diff(np.flip(non0_traj_fluor, 0)) / mean_fluct, height=peak_h)[0]) - 2
    )
    inds = np.unique(np.concatenate((ind1, ind2)))

    abn_t = non0_traj_t[inds]
    abn_inds = np.where(np.in1d(traj_t, abn_t))[0]  # find index of abn_t in traj_t
    return abn_inds


def count_num_fluor_pca_cord(cells, fluor_name, fluor_feature_name):
    for single_cell in cells:
        if hasattr(single_cell, fluor_name + "_feature_values"):
            print("this cell has pca cord")
            num_fluor_pca_cord = len(single_cell.vimentin_haralick_pca_cord)
            print(num_fluor_pca_cord)
            break
    return num_fluor_pca_cord
