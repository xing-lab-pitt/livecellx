from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
from livecellx.preprocess.utils import normalize_img_to_uint8
from livecellx.core.utils import gray_img_to_rgb, rgb_img_to_gray, label_mask_to_edt_mask
from livecellx.core import SingleCellTrajectory, SingleCellStatic
from livecellx.core.parallel import parallelize
from livecellx.livecell_logger import main_warning, main_info


def gen_mp4_from_frames(video_frames, output_file, fps):
    # Define the output video file name and properties
    frame_size = video_frames[0].shape[:2][::-1]  # reverse the order of width and height

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_file), fourcc, fps, frame_size)
    # Write each frame to the output video
    for frame in video_frames:
        out.write(frame)
    out.release()


def gen_mp4s_helper(sample, output_file, mask_output_file, combined_output_file, fps, padding_pixels):
    video_frames, video_frame_masks = video_frames_and_masks_from_sample(sample, padding_pixels=padding_pixels)
    combined_frames = combine_video_frames_and_masks(video_frames, video_frame_masks)
    # print("mask output file: ", mask_output_file)
    # print("combined output file: ", combined_output_file)
    # print("output file: ", output_file)
    # print("len video_frames: ", len(video_frames))
    gen_mp4_from_frames(video_frames, output_file, fps=fps)
    gen_mp4_from_frames(video_frame_masks, mask_output_file, fps=fps)
    gen_mp4_from_frames(combined_frames, combined_output_file, fps=fps)
    return output_file, mask_output_file, combined_output_file


def gen_samples_mp4s(
    sc_samples: List[List[SingleCellStatic]],
    samples_info_list,
    class_label,
    output_dir,
    fps=3,
    padding_pixels=50,
    prefix="",
):
    """
    Generate mp4 videos and masks from a list of SingleCellStatic samples.
    Args:
        sc_samples: A list of SingleCellStatic samples.
        sample_info_list: A list of dictionaries containing the information of the samples.
        class_label: A string representing the class label of the samples.
        output_dir: A Path object representing the directory to save the generated videos and masks.
        fps: An integer representing the frames per second of the generated videos.
        padding_pixels: An integer representing the number of pixels to pad around the cells in the generated videos and masks.
    Returns:
        A dictionary containing the file paths of the generated videos, masks, and combined videos.
    """
    type2paths = {"video": [], "mask": [], "combined": []}
    res_extra_info = []
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    # create output dir if not exist
    output_dir.mkdir(exist_ok=True, parents=True)
    helper_input_args = []
    for i, sample in enumerate(sc_samples):
        extra_sample_info = samples_info_list[i]
        if len(sample) == 0:
            main_warning(f"No cells in the sample {i} with extra info {extra_sample_info}, skipping...")
            continue
        if "tid" in extra_sample_info:
            output_file = output_dir / (
                f"{prefix}_tid-{int(extra_sample_info['tid'])}_{class_label}_sample-{i}_raw_padding-{padding_pixels}.mp4"
            )
            mask_output_file = output_dir / (
                f"{prefix}_tid-{int(extra_sample_info['tid'])}_{class_label}_sample-{i}_mask_padding-{padding_pixels}.mp4"
            )
            combined_output_file = output_dir / (
                f"{prefix}_tid-{int(extra_sample_info['tid'])}_{class_label}_sample-{i}_combined_padding-{padding_pixels}.mp4"
            )
        else:
            output_file = output_dir / (f"{prefix}_{class_label}_{i}_raw_padding-{padding_pixels}.mp4")
            mask_output_file = output_dir / (f"{prefix}_{class_label}_{i}_mask_padding-{padding_pixels}.mp4")
            combined_output_file = output_dir / (f"{prefix}_{class_label}_{i}_combined_padding-{padding_pixels}.mp4")
        helper_input_args.append((sample, output_file, mask_output_file, combined_output_file, fps, padding_pixels))
        type2paths["video"].append(output_file)
        type2paths["mask"].append(mask_output_file)
        type2paths["combined"].append(combined_output_file)

        res_extra_info.append(extra_sample_info)
    parallelize(gen_mp4s_helper, helper_input_args)
    return type2paths, res_extra_info


def gen_class2sample_samples(
    class2samples,
    class2sample_extra_info,
    data_dir,
    class_labels,
    padding_pixels,
    fps,
    frame_types=["video", "mask", "combined"],
    prefix="",
) -> pd.DataFrame:
    df_cols = ["path", "label_index", "padding_pixels", "frame_type", "src_dir", "track_id", "start_time", "end_time"]
    sample_info_df = pd.DataFrame(columns=df_cols)
    for class_label in class_labels:
        output_dir = Path(data_dir) / "videos"
        output_dir.mkdir(exist_ok=True, parents=True)
        video_frames_samples = class2samples[class_label]
        video_frames_samples_info = class2sample_extra_info[class_label]
        for padding_pixel in padding_pixels:
            frametype2paths, res_extra_info = gen_samples_mp4s(
                video_frames_samples,
                video_frames_samples_info,
                class_label,
                output_dir,
                padding_pixels=padding_pixel,
                fps=fps,
                prefix=prefix,
            )
            for selected_frame_type in frame_types:
                # mmaction_df = mmaction_df.append(pd.DataFrame([(str(path.name), class_labels.index(class_label), padding_pixel, selected_frame_type) for path in res_paths[selected_frame_type]], columns=["path", "label_index", "padding_pixels", "frame_type"]), ignore_index=True)
                sample_info_df = pd.concat(
                    [
                        sample_info_df,
                        pd.DataFrame(
                            [
                                (
                                    str(path.name),
                                    class_labels.index(class_label),
                                    padding_pixel,
                                    selected_frame_type,
                                    res_extra_info[i]["src_dir"] if "src_dir" in res_extra_info[i] else "",
                                    int(res_extra_info[i]["tid"]) if "tid" in res_extra_info[i] else -1,
                                    int(res_extra_info[i]["start_time"]) if "start_time" in res_extra_info[i] else -1,
                                    int(res_extra_info[i]["end_time"]) if "end_time" in res_extra_info[i] else -1,
                                )
                                for i, path in enumerate(frametype2paths[selected_frame_type])
                            ],
                            columns=df_cols,
                        ),
                    ]
                )
    return sample_info_df


def video_frames_and_masks_from_sample(
    sample: List[SingleCellStatic], padding_pixels=0
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

    if len(sample_scs) == 0:
        main_warning("No cells in the sample, skipping...")
        return [], []

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
        merged_label_mask = None
        tmp_img = scs_at_t[0].get_img_crop(bbox=largest_bbox, padding=padding_pixels)
        tmp_img = normalize_img_to_uint8(tmp_img)
        tmp_img = gray_img_to_rgb(tmp_img)
        for idx, sc in enumerate(scs_at_t):
            sc_label = idx + 1
            sc_mask = sc.get_sc_mask(bbox=largest_bbox, dtype=int, padding=padding_pixels)

            if merged_label_mask is None:
                merged_label_mask = np.zeros(sc_mask.shape, dtype=int)

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


def combine_video_frames_and_masks(video_frames, video_frame_masks, edt_transform=True):
    """returns a list of combined video frames and masks, each item contains a 3-channel image with first channel as frame and second channel as mask"""
    if edt_transform:
        video_frame_masks = [label_mask_to_edt_mask(x) for x in video_frame_masks]

    res_frames = []
    for frame, mask in zip(video_frames, video_frame_masks):
        frame = rgb_img_to_gray(frame)
        mask = rgb_img_to_gray(mask)
        res_frame = np.array([frame, mask, mask]).transpose(1, 2, 0)
        res_frames.append(res_frame)
    return res_frames
