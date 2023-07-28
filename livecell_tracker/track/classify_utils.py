from pathlib import Path
import numpy as np
from typing import List, Tuple

from scipy import ndimage
from livecell_tracker.core.single_cell import SingleCellStatic, SingleCellTrajectoryCollection
from livecell_tracker.core.utils import gray_img_to_rgb, rgb_img_to_gray, label_mask_to_edt_mask
from livecell_tracker.preprocess.utils import normalize_img_to_uint8
from livecell_tracker.core.sc_video_utils import (
    gen_class2sample_samples,
    video_frames_and_masks_from_sample,
    combine_video_frames_and_masks,
)


def gen_one_sc_samples_by_window(sctc: SingleCellTrajectoryCollection, window_size=7, step_size=1):
    tid2samples = {}
    for tid, sct in sctc:
        sct_samples = []
        sorted_scs = sct.get_sorted_scs()
        for i in range(0, len(sorted_scs) - window_size + 1, step_size):
            samples = sorted_scs[i : i + window_size]
            sct_samples.append(samples)
        tid2samples[tid] = sct_samples
    return tid2samples


def gen_inference_sctc_sample_videos(
    sctc: SingleCellTrajectoryCollection,
    class_label="unknown",
    window_size=7,
    step_size=1,
    prefix="",
    out_dir="tmp_sctc_samples",
    padding_pixels=[20],
    fps=3,
):
    sc_samples = []
    samples_info_list = []

    tid2samples = gen_one_sc_samples_by_window(sctc, window_size=window_size, step_size=step_size)
    for tid, samples in tid2samples.items():
        for i, sample in enumerate(samples):
            sc_samples.append(sample)
            samples_info_list.append({"tid": tid, "sample_idx": i})

    saved_sample_info_df = gen_class2sample_samples(
        {class_label: sc_samples},
        {class_label: samples_info_list},
        class_labels=[class_label],
        padding_pixels=padding_pixels,
        data_dir=out_dir,
        prefix=prefix,
        fps=fps,
    )
    return saved_sample_info_df
