import glob
import os
import pickle
import shutil
from datetime import datetime
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
from skimage.io import imread


# TODO: rename function
def correct_folder_str(folder):
    """Verify if the folder string is ended with '/'"""
    if folder[-1] != "/":
        folder = folder + "/"
    return folder


def count_pattern_in_folder(folder, pattern="*"):
    """How many files in the folder"""
    if folder[-1] != "/":
        folder = folder + "/"
    file_list = sorted(glob.glob(folder + "*" + pattern + "*"))
    print("%s " % folder + "has %s files" % len(file_list))
    return file_list


def create_folder(folder):
    """Create a folder. If the folder exist, erase and re-create."""
    folder = correct_folder_str(folder)

    if os.path.exists(folder):  # recreate folder every time.
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)
    print("%s folder is created. \n" % folder)

    return folder


def copy_file(start_folder, pattern, num_of_files, end_folder):
    """
    Copy a list of files to destinated folder.
    """
    file_list = count_pattern_in_folder(start_folder, pattern)[:num_of_files]

    if end_folder[-1] != "/":
        end_folder = end_folder + "/"

    if os.path.exists(end_folder):  # recreate folder every time.
        shutil.rmtree(end_folder)
        os.makedirs(end_folder)
    else:
        os.makedirs(end_folder)

    for f in file_list:
        shutil.copy(f, end_folder)

    print("The copy processes have completed.")


def folder_space_replace(parent):
    """
    Given a folder, replace all space in the path for subfiles and subfolders.
    parent - string, top level folder path.
    """
    for path, folders, files in os.walk(parent):
        for f in files:
            os.rename(os.path.join(path, f), os.path.join(path, f.replace(" ", "_")))
        for i in range(len(folders)):
            new_name = folders[i].replace(" ", "_")
            os.rename(os.path.join(path, folders[i]), os.path.join(path, new_name))
            folders[i] = new_name


# move selected number of files into another folder
def migrate_img_to_folder(img_folder, aim_folder, num):

    """
    Migrate a select number of pictures to a aim_folder

    img_folder - string, starting folder path
    aim_folder - string
    num - int, number of images to migrate"""

    img_folder = correct_folder_str(img_folder)
    aim_folder = correct_folder_str(aim_folder)
    create_folder(aim_folder)

    img_files = count_pattern_in_folder(img_folder)
    ins_list = np.int32(np.linspace(0, len(img_files), num)[:-1])

    img_files = list_select_by_index(img_files, ins_list)
    i = 0
    while i < len(img_files):
        name = os.path.basename(img_files[i])
        name = aim_folder + name
        shutil.copy(img_files[i], aim_folder)

        i = i + 1
    print("The files are migrated.")


# ------------------- Image Show --------------------


def folder_img_show(folder, num, gray=1):

    """
    A convient function for print pictures under a folder.
    5 pics per row.
    num - int, can limiting the number of frame printing
    gray - string, for color map. default gray
    """
    fig = plt.figure(figsize=(20, 20))
    img_path_list = sorted(glob.glob(folder + "/*"))

    # plot
    # each row have 5 figures
    img_num = len(img_path_list)
    num = min(img_num, num)
    rows = int(num / 5.0) + 1

    i = 0
    while i < num:
        fig.add_subplot(rows, 5, i + 1)
        if gray:
            plt.imshow(imread(img_path_list[i]) / 256.0, cmap=gray)
        else:
            plt.imshow(imread(img_path_list[i]) / 256.0, cmap="gray")
        i = i + 1

    plt.show()


def folder_img_edge_show(folder, num, gray=0):

    """
    A convient function for print pictures under a folder.
    5 pics per row.
    num - int, can limiting the number of frame printing
    gray - string, for color map. default gray
    """
    fig = plt.figure(figsize=(20, 20))
    img_path_list = sorted(glob.glob(folder + "/*"))

    # plot
    # each row have 5 figures
    img_num = len(img_path_list)
    num = min(img_num, num)
    rows = int(num / 5.0) + 1

    i = 0
    while i < num:
        fig.add_subplot(rows, 5, i + 1)
        if gray:
            plt.imshow(filters.roberts(imread(img_path_list[i])) / 256.0, cmap=gray)
        else:
            plt.imshow(filters.roberts(imread(img_path_list[i])) / 256.0, cmap="gray")
        i = i + 1

    plt.show()


def list2d_plot_show(list2d, num, if_limit=0):

    """
    A convient function for printing pictures for a 2d list of numbers.
    5 pics per row.
    num - int, can limiting the number of frame printing
    """
    fig = plt.figure(figsize=(20, 20))

    # plot
    # each row have 5 figures
    img_num = len(list2d)
    num = min(img_num, num)
    rows = int(num / 5.0) + 1

    i = 0
    while i < num:
        fig.add_subplot(rows, 5, i + 1)
        plt.plot(list2d[i])
        if if_limit:
            pass
        else:
            plt.ylim((0, 100))
        i = i + 1

    plt.show()


def list_histimg_show(list_2d, bin_num, fig_num, w_num, h_num, w=20, h=20):

    """
    A convient function for printing histogram for a 2d list.
    list_2d - a list of list of number. For each number list, plot a histograph.
    fig_num - int, should be smaller than dimention of list_2d.
    """
    flat_list = np.log(np.array(list(chain.from_iterable(list_2d))) + 1)
    hist, bin_edges = np.histogram(flat_list, bin_num)

    fig = plt.figure(figsize=(w, h))
    if len(list_2d) < fig_num:
        print("Not enough pictures to print")
        return 1

    for i, ls in enumerate(list_2d[:fig_num]):
        fig.add_subplot(w_num, h_num, i + 1)
        plt.hist(np.log(np.array(ls) + 1), bin_edges)
    plt.show()


def list_label_show(label_list, fig_num, w_num, h_num, costom_cmap="coolwarm", w=20, h=20):
    """
    A convient function for printing label images
    """
    fig = plt.figure(figsize=(w, h))
    if len(label_list) < fig_num:
        print("Not enough pictures to print")
        return 1

    for i, label in enumerate(label_list[:fig_num]):
        fig.add_subplot(w_num, h_num, i + 1)
        plt.imshow(label, cmap=costom_cmap)
    plt.show()


# def img_mask_show(img


# -------------------------- General ------------------------------


def list_select_by_index(the_list, index_list):
    """Return a list of selected items."""
    selected_elements = [the_list[index] for index in index_list]
    return selected_elements


def print_time():

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


# -------------------------- Migrating files for Incucyte ------------------------------


def folder_pattern(folder, start_posi, end_posi):
    """
    Return a list of patterns.
    Notice the actual index is (start - 1, end)
    """
    img_files = count_pattern_in_folder(folder)
    pattern_list = []
    i = 0
    while i < len(img_files):
        base_name = os.path.basename(img_files[i])
        pattern = base_name[int(start_posi - 1) : int(end_posi)]
        pattern_list.append(pattern)

        i = i + 1
    pattern_list = np.unique(pattern_list)
    return pattern_list


def pattern_polish(p):
    """if the pattern ends up with _, remove it."""
    if p[-1] == "_":
        p = p[:-2] + "0" + p[-2]
    return p


def to_subfolder(start_folder, end_folder, pattern, num_of_files=-1):
    """
    moving files in starting_folder to end_folder according to patterns
    """
    start_folder = correct_folder_str(start_folder)
    end_folder = correct_folder_str(end_folder)
    create_folder(end_folder)
    i = 0
    while i < len(pattern):

        p = pattern[i]  # current pattern
        p_name = pattern_polish(p)  # folder name without _
        d = correct_folder_str(end_folder) + p_name  # current folder
        create_folder(d)
        copy_file(start_folder, p, num_of_files, d)
        i = i + 1
    print("File move completed, please check your folder")


def incucyte_folder_sort(folder, pattern_start_ind, pattern_end_ind, num_of_files=-1):
    """Function for moving files from an incucyte folder."""
    p_list = folder_pattern(folder, pattern_start_ind, pattern_end_ind)
    print(p_list)
    aim_folder = correct_folder_str(folder)[:-1] + "-sorted"
    to_subfolder(folder, aim_folder, p_list, num_of_files)

    print("The processes is done")


# -------------------------- Icnn training ploting ------------------------------
def icnn_am_history_plot(am_hist_file, if_save=0):

    history = pickle.load(open(am_hist_file, "rb"))
    # print(history.keys())

    fig, ax1 = plt.subplots()

    color = "midnightblue"
    ax1.plot(history["accuracy"], color=color)
    ax1.set_ylabel("accuracy", color=color)
    ax1.set_xlabel("epoch")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:orange"

    ax2.plot(history["loss"], color=color)
    ax2.set_ylabel("loss", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    if if_save:
        plt.savefig(am_hist_file[:-4] + ".pdf", dpi=200)

    plt.show()
