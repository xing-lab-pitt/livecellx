# In[0]: import
import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[1]: main
def main(crop):

    # gives the cols and rows where they are all zero (crop.any(axis=1) returns a [1 x number of rows] array
    # that has True if the row contained a number != 0. The ~ just flips it so rows with all zeros become True
    # instead of False. Then np.where finds all values of True (i.e. all rows with all zeros in it). Then for some
    # reason np.where returns a nested list so I just grabbed the list of row numbers that contained all zeros from
    # within)
    blank_rows = np.where(~crop.any(axis=1))[0]
    blank_cols = np.where(~crop.any(axis=0))[0]

    # deletes empty rows/columns
    crop = np.delete(crop, blank_rows, axis=0)
    crop = np.delete(crop, blank_cols, axis=1)

    # This checks to see if after removing all rows/columns with all zeros, are there any more zeros left
    removed_rows = list(blank_rows.copy())
    removed_cols = list(blank_cols.copy())
    while len(np.where(crop == 0)[0]):

        # np.argwhere returns a list of lists of XY coords ([[X1 Y1][X2 Y2]]) so I grab the first set of XY coords here
        zero_coord = np.argwhere(crop == 0)[0]

        # I then figure out if there is more zeros in the row vs the column. I do this with a similar logic as
        # shown above. I get the row/column of the crop array (already full rows/columns of zeros are removed)
        # that has a zero in it. Then I convert this single column/row to a bool, flip the booleans with ~ and
        # and then sum them up so I get a count of zeros per row/column
        zeros_row = np.sum(~crop[zero_coord[0]].astype(bool))
        zeros_col = np.sum(~crop[:, zero_coord[1]].astype(bool))

        # Depending on whether there are more zeros in the column or row, I remove the one with the highest
        # num of zeros. There are siutations where this will break, if a row has 5 zeros, and the column has 2,
        # but the row is in the middle of the picture and instead of removing 4 columns from each side of the picture,
        # it could just remove many rows from the middle of the picture. From what I have seen, this is never the case
        # , but it could happen
        if zeros_col > zeros_row:
            crop = np.delete(crop, zero_coord[1], axis=1)
            removed_cols.append(zero_coord[1])
        else:
            crop = np.delete(crop, zero_coord[0], axis=0)
            removed_rows.append(zero_coord[0])
    return crop, removed_cols, removed_rows
