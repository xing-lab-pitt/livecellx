import glob
from pathlib import Path
import numpy as np
from typing import List, Tuple

from scipy import ndimage
from livecellx.core.single_cell import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection
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


def gen_tid2samples_by_window(sctc: SingleCellTrajectoryCollection, window_size=7, step_size=1):
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

    tid2samples, tid2start_end_times = gen_tid2samples_by_window(sctc, window_size=window_size, step_size=step_size)
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


def save_data_input(data_input, file_path):
    from livecellx.core.sc_video_utils import gen_mp4_from_frames

    imgs = data_input[1][2].detach().cpu().numpy()  # 8 x 224 x 224
    masks = data_input[1][0].detach().cpu().numpy()  # 8 x 224 x 224
    imgs = list(imgs)
    masks = list(masks)
    imgs = [normalize_img_to_uint8(img) for img in imgs]
    masks = [normalize_img_to_uint8(mask) for mask in masks]

    # already edt transformed
    frames = combine_video_frames_and_masks(imgs, masks, is_gray=True, edt_transform=False)
    gen_mp4_from_frames(frames, file_path)


def is_decord_invalid_video(path):
    """More information: https://github.com/dmlc/decord/issues/150"""
    import decord

    reader = decord.VideoReader(str(path))
    reader.seek(0)
    imgs = list()
    frame_inds = range(0, len(reader))
    for idx in frame_inds:
        reader.seek(idx)
        frame = reader.next()
        imgs.append(frame.asnumpy())
        frame = frame.asnumpy()

        num_channels = frame.shape[-1]
        if num_channels != 3:
            print("invalid video for decord (https://github.com/dmlc/decord/issues/150): ", path)
            return True
        # fig, axes = plt.subplots(1, num_channels, figsize=(20, 10))
        # for i in range(num_channels):
        #     axes[i].imshow(frame[:, :, i])
        # plt.show()
    return False


def insert_time_segments(new_segment: Tuple[int, int], disjoint_segments: list):
    """add the new segment to segments, merge if there is overlap between new_segment and any segment in segments, keep all segments non-overlapping"""
    if len(disjoint_segments) == 0:
        disjoint_segments.append(new_segment)
        return
    # find the first segment that overlaps with new_segment
    merged = False
    for i, segment in enumerate(disjoint_segments):
        if segment[0] <= new_segment[0] <= segment[1] or segment[0] <= new_segment[1] <= segment[1]:
            # merge the new segment with the segment
            disjoint_segments[i] = (min(segment[0], new_segment[0]), max(segment[1], new_segment[1]))
            merged = True
    # no overlap found, add the new segment
    if not merged:
        disjoint_segments.append(new_segment)
        disjoint_segments.sort(key=lambda x: x[0])
        return
    # check if there is any overlap between segments
    i = 0
    while i < len(disjoint_segments) - 1:
        if disjoint_segments[i][1] >= disjoint_segments[i + 1][0]:
            # merge the two segments
            disjoint_segments[i] = (disjoint_segments[i][0], max(disjoint_segments[i][1], disjoint_segments[i + 1][1]))
            disjoint_segments.pop(i + 1)
        else:  # no overlap
            i += 1
    return disjoint_segments


def infer_sliding_window_traj(
    sct: SingleCellTrajectory,
    model,
    frame_type,
    window_size=8,
    padding_pixels=[200],
    out_dir="./_tmp_samples",
    prefix="samples",
    fps=3,
    class_labels=[0],
    class_names=["mitosis"],
):
    """
    Infers the sliding window trajectory for a given SingleCellTrajectory object using a pre-trained model.

    Args:
        sct (SingleCellTrajectory): The SingleCellTrajectory object to infer the sliding window trajectory for.
        model: The pre-trained model to use for inference.
        frame_type: The type of frame to use for inference.
        window_size (int): The size of the sliding window to use for generating samples.
        padding_pixels (list): The number of pixels to pad the samples with.
        out_dir (str): The output directory to save the generated samples to.
        prefix (str): The prefix to use for the generated sample filenames.
        fps (int): The frames per second to use for the generated sample videos.
        class_labels (list): The class labels to use for inference.
        class_names (list): The class names to use for inference.

    Returns:
        dict: A dictionary containing the disjoint segments, sample output directory, all video dataframe, test dataframe, and predicted labels.
    """
    from tqdm import tqdm
    from mmaction.apis import init_recognizer, inference_recognizer

    # Create a temporary SingleCellTrajectoryCollection and add the trajectory to it
    tmp_sctc = SingleCellTrajectoryCollection()
    tmp_sctc.add_trajectory(sct)

    # Generate the samples for the trajectory using a sliding window approach
    tid2samples, tid2start_end_times = gen_tid2samples_by_window(tmp_sctc, window_size=window_size)

    # Get the samples for the specific trajectory
    sample_tid_samples = tid2samples[sct.track_id]

    # Generate the inference samples and save them to a video
    _sample_output_dir = Path(out_dir)
    specific_traj_video_df = gen_inference_sctc_sample_videos(
        tmp_sctc, padding_pixels=padding_pixels, out_dir=_sample_output_dir, prefix=prefix, fps=fps
    )

    if class_labels is not None:
        assert len(class_labels) == len(class_names), "save_classes and class_names must have the same length"
        class_name_to_segments = {}
        save_class_dir = _sample_output_dir / "classes_videos"
        class2dir = {}
        for class_name in class_names:
            if not save_class_dir.exists():
                save_class_dir.mkdir(parents=True)
            class_dir = save_class_dir / class_name
            if not class_dir.exists():
                class_dir.mkdir(parents=True)
            class2dir[class_name] = class_dir
            class_name_to_segments[class_name] = []
        class_label2name = {class_labels[i]: class_name for i, class_name in enumerate(class_names)}

    selected_df = specific_traj_video_df[specific_traj_video_df["frame_type"] == frame_type]

    preds = []
    for i, row in tqdm(selected_df.iterrows(), total=len(selected_df)):
        video_filename = row["path"]
        video_path = str(_sample_output_dir / "videos" / video_filename)

        # Inference
        results = inference_recognizer(model, video_path)
        if "pred_label" in results.keys():
            # TimeSformer
            predicted_label = results.pred_label.cpu().numpy()[0]
        else:
            # TSN
            predicted_label = results.pred_labels.item.cpu().numpy()[0]

        if predicted_label in class_labels:
            pred_class_dir = class2dir[class_names[predicted_label]]
            import shutil

            shutil.copy(video_path, str(pred_class_dir / video_filename))
            class_name = class_label2name[predicted_label]
            insert_time_segments((row["start_time"], row["end_time"]), class_name_to_segments[class_name])

        preds.append(predicted_label)

    return {
        "disjoint_segments": class_name_to_segments,
        "sample_output_dir": _sample_output_dir,
        "all_video_df": specific_traj_video_df,
        "test_df": selected_df,
        "preds": preds,
    }
