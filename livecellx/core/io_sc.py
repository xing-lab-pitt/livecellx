import tqdm
import numpy as np
from multiprocessing import Pool
from skimage.measure import regionprops, find_contours
from livecellx.livecell_logger import main_info, main_warning
from livecellx.segment.ou_simulator import find_contours_opencv
from livecellx.core.single_cell import SingleCellStatic
from livecellx.core.sc_key_manager import SingleCellMetaKeyManager


def process_scs_from_single_label_mask(
    label_mask, img_dataset, time, bg_val=0, min_contour_len=10, label_mask_dataset=None, verbose=False
):
    labels = set(np.unique(label_mask))
    if bg_val in labels:
        labels.remove(bg_val)
    contours = []
    labels = list(labels)
    contour_labels = []
    for label in labels:
        bin_mask = (label_mask == label).astype(np.uint8)
        label_contours = find_contours_opencv(bin_mask)
        # assert len(label_contours) == 1, "at time {}, label {} has {} contours".format(time, label, len(label_contours))
        # contours.append(label_contours[0])

        # warn the users
        filtered_label_contours = []
        for contour in label_contours:
            if len(contour) >= min_contour_len:
                filtered_label_contours.append(contour)

        if verbose and len(filtered_label_contours) < len(label_contours):
            main_info(
                "at time {}, label {} has {} contours found by opencv, {} of them are filtered by contour length threshold: {}".format(
                    time,
                    label,
                    len(label_contours),
                    len(label_contours) - len(filtered_label_contours),
                    min_contour_len,
                )
            )

        label_contours = filtered_label_contours

        if verbose and len(label_contours) > 1:
            main_warning("at time {}, label {} has {} contours".format(time, label, len(label_contours)))
            main_warning("lengths of each contour: {}".format([len(c) for c in label_contours]))

        contours.extend(label_contours)
        contour_labels.extend([label] * len(label_contours))

    # contours = find_contours(seg_mask) # skimage: find_contours
    _scs = []
    for i, contour in enumerate(contours):
        label = int(
            contour_labels[i]
        )  # int important here to get rid of numpy.int64 or numpy.int8, etc, to avoid json dump error
        sc = SingleCellStatic(
            timeframe=time,
            img_dataset=img_dataset,
            mask_dataset=label_mask_dataset,
            contour=contour,
        )
        sc.meta[SingleCellMetaKeyManager.MASK_LABEL] = label
        _scs.append(sc)
    return _scs


# TODO: fix the function below
def process_scs_from_label_mask_dataset(label_mask_dataset, img_dataset, time, bg_val=0, min_contour_len=10):
    """process single cells from one label mask. Store labels of single cells in their meta data.

    Parameters
    ----------
    label_mask_dataset : _type_
        _description_
    dic_dataset : _type_
        _description_
    time : _type_
        _description_
    bg_val : int, optional
        _description_, by default 0

    Returns
    -------
    _type_
        _description_
    """

    return process_scs_from_single_label_mask(
        label_mask_dataset.get_img_by_time(time),
        img_dataset,
        time,
        bg_val=bg_val,
        min_contour_len=min_contour_len,
        label_mask_dataset=label_mask_dataset,
    )


def process_mask_wrapper(args):
    return process_scs_from_label_mask_dataset(*args)


# TODO: use parallelize function in the future
def prep_scs_from_mask_dataset(mask_dataset, img_dataset, cores=None):
    scs = []
    inputs = [(mask_dataset, img_dataset, time) for time in mask_dataset.time2url.keys()]
    pool = Pool(processes=cores)
    for _scs in tqdm.tqdm(pool.imap_unordered(process_mask_wrapper, inputs), total=len(inputs)):
        scs.extend(_scs)
    pool.close()
    pool.join()
    return scs
