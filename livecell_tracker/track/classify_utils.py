import numpy as np
from typing import List, Tuple
from livecell_tracker.core.single_cell import SingleCellStatic
from livecell_tracker.core.utils import gray_img_to_rgb
from livecell_tracker.preprocess.utils import normalize_img_to_uint8


def video_frames_and_masks_from_sample(sample: List[SingleCellStatic]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Given a sample of SingleCell objects, returns a list of video frames and a list of video frame masks.
    Each video frame is a numpy array representing an RGB image of the cells in the sample at a particular timepoint.
    Each video frame mask is a numpy array representing a grayscale image of the cells in the sample at a particular timepoint,
    where each cell is labeled with a unique integer value.

    Args:
    - sample: a list of SingleCell objects representing a sample of cells to be included in the video.

    Returns:
    - video_frames: a list of numpy arrays representing RGB images of the cells in the sample at each timepoint.
    - video_frame_masks: a list of numpy arrays representing grayscale images of the cells in the sample at each timepoint,
    where each cell is labeled with a unique integer value.
    """
    scs_by_time = {}
    for sc in sample:
        if sc.timeframe not in scs_by_time:
            scs_by_time[sc.timeframe] = []
        scs_by_time[sc.timeframe].append(sc)
    sc_times = sorted(scs_by_time.keys())
    sample_scs = []
    for i in range(0, len(sc_times)):
        time = sc_times[i]
        scs_at_time = scs_by_time[time]
        sample_scs.append(scs_at_time)

    # get the largest bounding box
    largest_bbox = [np.inf, np.inf, -np.inf, -np.inf]
    for scs_at_t in sample_scs:
        for sc in scs_at_t:
            bbox = sc.bbox
            if bbox[0] < largest_bbox[0]:
                largest_bbox[0] = bbox[0]
            if bbox[1] < largest_bbox[1]:
                largest_bbox[1] = bbox[1]
            if bbox[2] > largest_bbox[2]:
                largest_bbox[2] = bbox[2]
            if bbox[3] > largest_bbox[3]:
                largest_bbox[3] = bbox[3]

    # make largest_bbox coords integer
    largest_bbox = [int(x) for x in largest_bbox]

    video_frames = []
    video_frame_masks = []
    for scs_at_t in sample_scs:
        merged_label_mask = np.zeros(
            (largest_bbox[2] - largest_bbox[0], largest_bbox[3] - largest_bbox[1]), dtype=np.uint8
        )
        tmp_img = scs_at_t[0].get_img_crop(bbox=largest_bbox)
        tmp_img = normalize_img_to_uint8(tmp_img)
        tmp_img = gray_img_to_rgb(tmp_img)
        for idx, sc in enumerate(scs_at_t):
            sc_label = idx + 1
            sc.bbox = largest_bbox
            sc_mask = sc.get_sc_mask(bbox=largest_bbox, dtype=int)

            # Warning: simply add the label masks will cause overlapping cells to generate unexpected labels
            _nonzero = sc_mask > 0
            merged_label_mask[_nonzero] = sc_label

            # # for debugging
            # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # axes[0].imshow(tmp_img)
            # axes[1].imshow(merged_label_mask)
            # plt.show()
        video_frames.append(tmp_img)
        video_frame_masks.append(gray_img_to_rgb(normalize_img_to_uint8(merged_label_mask)))
    return video_frames, video_frame_masks
