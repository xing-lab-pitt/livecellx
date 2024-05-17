import glob
import os
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from livecellx.core.io_utils import save_png
from livecellx.core.single_cell import largest_bbox
from livecellx.preprocess.utils import overlay, overlay_by_color


def plot_sctc(
    filtered_sctc, out_dir, prefix="normal", save_whole=False, save_masked=True, save_sc_traj=True, padding=50, dpi=300
):
    """
    Plot the single-cell trajectories and save the plots as images.

    Args:
        filtered_sctc (FilteredSCTC): The filtered single-cell trajectories.
        out_dir (str): The output directory to save the plots.
        prefix (str, optional): The prefix to be added to the saved plot filenames. Defaults to "normal".
        save_whole (bool, optional): Whether to save the whole single-cell images. Defaults to False.
        save_masked (bool, optional): Whether to save the masked single-cell images. Defaults to True.
        save_sc_traj (bool, optional): Whether to save the single-cell trajectories. Defaults to True.
        padding (int, optional): The padding around the single-cell images. Defaults to 50.
    """

    for sct in filtered_sctc.get_all_trajectories():
        tid = sct.track_id
        bbox = largest_bbox(sct.get_all_scs())
        # Convert to int
        bbox = [int(b) for b in bbox]

        # Create subplots
        img_crops = []
        overlayed_imgs = []
        for idx, time in enumerate(sct.times):
            if time not in sct.times:
                # Blank subplot for missing time points
                axes[idx].axis("off")
                continue

            sc_img = sct[time].get_img_crop(bbox=bbox, padding=padding)  # Get the cropped image

            sc_mask_crop = sct[time].get_contour_mask(bbox=bbox, padding=padding, dtype=np.uint8)

            # Save the image as PNG
            if save_whole:
                save_png(out_dir / f"{prefix}_whole_{tid}_{time}.png", sc_img)
            # overlayed = overlay(sc_img, sc_img_crop, mask_channel_rgb_val=100, img_channel_rgb_val_factor=1)
            # Draw the image on the plot
            # axes[idx].imshow(sc_img, cmap='gray')
            # axes[idx].axis('off')
            overlayed = overlay_by_color(sc_img, sc_mask_crop, color=(10, 10, 0), alpha=0.05)

            # fig, axes = plt.subplots(1, 3, figsize=(6, 3), dpi=300)
            # axes[0].imshow(sc_mask_crop)

            # axes[1].imshow(sc_img)
            # axes[2].imshow(overlayed)
            # plt.show()
            img_crops.append(sc_img)
            overlayed_imgs.append(overlayed)

        if not save_sc_traj:
            # Adjust layout and save the figure
            fig, axes = plt.subplots(1, len(sct), figsize=(3 * len(sct), 4), dpi=dpi)
            if len(sct) == 1:
                axes = [axes]
            for idx, img in enumerate(img_crops):
                axes[idx].imshow(img, cmap="gray")
                axes[idx].axis("off")
            fig.tight_layout(w_pad=0.2)
            plt.savefig(out_dir / f"{prefix}_{tid}.png")
            plt.close(fig)

        if save_masked:
            fig, axes = plt.subplots(1, len(sct), figsize=(3 * len(sct), 4), dpi=dpi)
            if len(sct) == 1:
                axes = [axes]
            for idx, img in enumerate(overlayed_imgs):
                axes[idx].imshow(img)
                axes[idx].axis("off")
            fig.tight_layout(w_pad=0.2)
            plt.savefig(out_dir / f"{prefix}_masked_{tid}.png")
            plt.close(fig)
    plt.close()
