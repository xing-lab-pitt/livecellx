from pathlib import Path
from typing import List
import cv2
import numpy as np
import pandas as pd
from livecell_tracker.core import SingleCellTrajectory, SingleCellStatic
from livecell_tracker.track.classify_utils import video_frames_and_masks_from_sample, combine_video_frames_and_masks
from livecell_tracker.core.parallel import parallelize


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


def gen_samples_df(
    class2samples, class2sample_extra_info, data_dir, class_labels, padding_pixels, frame_types, fps, prefix=""
):
    df_cols = ["path", "label_index", "padding_pixels", "frame_type", "src_dir", "track_id"]
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
                                )
                                for i, path in enumerate(frametype2paths[selected_frame_type])
                            ],
                            columns=df_cols,
                        ),
                    ]
                )
    return sample_info_df
