from typing import List
import cv2
import numpy as np
import pandas as p

d
from livecell_tracker.core import SingleCellTrajectory, SingleCellStatic
from livecell_tracker.track.classify_utils import video_frames_and_masks_from_sample, combine_video_frames_and_masks


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
    res_paths = {"video": [], "mask": [], "combined": []}
    res_extra_info = []
    for i, sample in enumerate(sc_samples):
        output_file = output_dir / (f"{prefix}_{class_label}_{i}_raw_padding-{padding_pixels}.mp4")
        mask_output_file = output_dir / (f"{prefix}_{class_label}_{i}_mask_padding-{padding_pixels}.mp4")
        combined_output_file = output_dir / (f"{prefix}_{class_label}_{i}_combined_padding-{padding_pixels}.mp4")

        # record video file path and class label
        video_frames, video_frame_masks = video_frames_and_masks_from_sample(sample, padding_pixels=padding_pixels)
        combined_frames = combine_video_frames_and_masks(video_frames, video_frame_masks)

        # # for debug
        # print("len video_frames: ", len(video_frames))
        # print("len masks video: ", len(video_frame_masks))
        # print("len combined_frames: ", len(combined_frames))

        gen_mp4_from_frames(video_frames, output_file, fps=fps)
        gen_mp4_from_frames(video_frame_masks, mask_output_file, fps=fps)
        gen_mp4_from_frames(combined_frames, combined_output_file, fps=fps)
        res_paths["video"].append(output_file)
        res_paths["mask"].append(mask_output_file)
        res_paths["combined"].append(combined_output_file)

        extra_sample_info = samples_info_list[i]
        res_extra_info.append(extra_sample_info)
    return res_paths, res_extra_info


def gen_samples_df(
    class2samples, class2sample_extra_info, data_dir, class_labels, padding_pixels, frame_types, fps, prefix=""
):
    df_cols = ["path", "label_index", "padding_pixels", "frame_type", "src_dir"]
    sample_info_df = pd.DataFrame(columns=df_cols)
    for class_label in class_labels:
        output_dir = Path(data_dir) / "videos"
        output_dir.mkdir(exist_ok=True, parents=True)
        video_frames_samples = class2samples[class_label]
        video_frames_samples_info = class2sample_extra_info[class_label]
        for padding_pixel in padding_pixels:
            res_paths, res_extra_info = gen_samples_mp4s(
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
                                    res_extra_info[i]["src_dir"],
                                )
                                for i, path in enumerate(res_paths[selected_frame_type])
                            ],
                            columns=df_cols,
                        ),
                    ]
                )
    return sample_info_df
