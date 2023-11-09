import glob
from pathlib import Path
import numpy as np
from typing import List, Tuple

from scipy import ndimage
from livecellx.core.single_cell import SingleCellStatic, SingleCellTrajectoryCollection
from livecellx.core.utils import gray_img_to_rgb, rgb_img_to_gray, label_mask_to_edt_mask
from livecellx.preprocess.utils import normalize_img_to_uint8
from livecellx.core.sc_video_utils import (
    gen_class2sample_samples,
    video_frames_and_masks_from_sample,
    combine_video_frames_and_masks,
)


def load_class2samples_from_json_dir(
    sample_json_dir: Path, class_subfolders=["mitosis", "apoptosis", "normal"]
) -> dict:
    # sample_paths = glob.glob(str(sample_json_dir / "*.json"))
    class2samples = {}
    for subfolder in class_subfolders:
        class2samples[subfolder] = []
        sample_paths = glob.glob(str(sample_json_dir / subfolder / "*.json"))
        for sample_path in sample_paths:
            sample = SingleCellStatic.load_single_cells_json(sample_path)
            for sc in sample:
                sc.meta["sample_src_dir"] = str(sample_json_dir)
                sc.meta["sample_src_class"] = str(subfolder)
            class2samples[subfolder].append(sample)
    return class2samples


def load_all_json_dirs(
    sample_json_dirs: Path, class_subfolders=["mitosis", "apoptosis", "normal"]
) -> tuple[dict[str, list[SingleCellStatic]], dict[str, list[dict]]]:
    all_class2samples = {}
    all_class2sample_extra_info = {}
    for sample_json_dir in sample_json_dirs:
        _class2samples = load_class2samples_from_json_dir(sample_json_dir, class_subfolders=class_subfolders)
        print(_class2samples)
        for class_name in _class2samples:
            # report how many samples loaded from the sample json dir
            print(f"Loaded {len(_class2samples[class_name])} annotated samples from {sample_json_dir / class_name}")

        for class_name in _class2samples:
            if class_name not in all_class2samples:
                all_class2samples[class_name] = _class2samples[class_name]
            else:
                all_class2samples[class_name] += _class2samples[class_name]

            _extra_info = [{"src_dir": sample_json_dir} for _ in range(len(_class2samples[class_name]))]
            if class_name not in all_class2sample_extra_info:
                all_class2sample_extra_info[class_name] = _extra_info
            else:
                all_class2sample_extra_info[class_name] += _extra_info
    return all_class2samples, all_class2sample_extra_info


def gen_one_sc_samples_by_window(sctc: SingleCellTrajectoryCollection, window_size=7, step_size=1):
    tid2samples = {}
    tid2start_end_times = {}
    for tid, sct in sctc:
        sct_samples = []
        start_end_times = []
        sorted_scs = sct.get_sorted_scs()
        for i in range(0, len(sorted_scs) - window_size + 1, step_size):
            samples = sorted_scs[i : i + window_size]
            sct_samples.append(samples)
            start_end_times.append((samples[0].timeframe, samples[-1].timeframe))
        tid2samples[tid] = sct_samples
        tid2start_end_times[tid] = start_end_times
    return tid2samples, tid2start_end_times


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

    tid2samples, tid2start_end_times = gen_one_sc_samples_by_window(sctc, window_size=window_size, step_size=step_size)
    for tid, samples in tid2samples.items():
        start_end_times = tid2start_end_times[tid]
        for i, sample in enumerate(samples):
            sc_samples.append(sample)
            samples_info_list.append(
                {"tid": tid, "sample_idx": i, "start_time": start_end_times[i][0], "end_time": start_end_times[i][1]}
            )

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
