from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from typing import List
from livecellx.core.single_cell import SingleCellStatic
from livecellx.livecell_logger import main_info, main_warning


def add_colorbar(im, ax, fig):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")


def crop_or_pad_img(img_crop, fix_dims):
    """Crop or pad a 2D image to fix_dims; For crop, crop from the central region"""
    if fix_dims is not None:
        if img_crop.shape[0] > fix_dims[0]:
            start = (img_crop.shape[0] - fix_dims[0]) // 2
            img_crop = img_crop[start : start + fix_dims[0], :]
        else:
            pad = (fix_dims[0] - img_crop.shape[0]) // 2
            img_crop = np.pad(img_crop, ((pad, pad), (0, 0)), mode="constant", constant_values=0)
        if img_crop.shape[1] > fix_dims[1]:
            start = (img_crop.shape[1] - fix_dims[1]) // 2
            img_crop = img_crop[:, start : start + fix_dims[1]]
        else:
            pad = (fix_dims[1] - img_crop.shape[1]) // 2
            img_crop = np.pad(img_crop, ((0, 0), (pad, pad)), mode="constant", constant_values=0)

    return img_crop


def crop_or_pad_img(img_crop, fix_dims):
    """Crop or pad a 2D image to fix_dims; For crop, crop from the central region"""
    if fix_dims is not None:
        if img_crop.shape[0] > fix_dims[0]:
            start = (img_crop.shape[0] - fix_dims[0]) // 2
            img_crop = img_crop[start : start + fix_dims[0], :]
        else:
            pad = (fix_dims[0] - img_crop.shape[0]) // 2
            img_crop = np.pad(img_crop, ((pad, pad), (0, 0)), mode="constant", constant_values=0)

        if img_crop.shape[1] > fix_dims[1]:
            start = (img_crop.shape[1] - fix_dims[1]) // 2
            img_crop = img_crop[:, start : start + fix_dims[1]]
        else:
            pad = (fix_dims[1] - img_crop.shape[1]) // 2
            img_crop = np.pad(img_crop, ((0, 0), (pad, pad)), mode="constant", constant_values=0)

    return img_crop


def viz_embedding_region(
    embedding,
    scs: List[SingleCellStatic],
    padding=40,
    x_range=(float("-inf"), float("inf")),
    y_range=(float("-inf"), float("inf")),
    title="Single cell in Embedding Space",
    max_crops=10,
    fix_dims=None,
    randomly_select=True,
    sort_by_x=True,
    dpi=300,
    show_mask=False,
):
    _crops = []
    mask_crops = []
    _crop_coords = []
    if fix_dims is not None:
        main_info(f"Setting: fix_dims={fix_dims}")

    indices = range(len(scs))
    if randomly_select:
        indices = np.random.choice(len(scs), len(scs), replace=False)

    for idx in indices:
        _sc = scs[idx]
        # print("embedding[idx]", embedding[idx])
        # Embedding in range
        if x_range[0] <= embedding[idx][0] <= x_range[1] and y_range[0] <= embedding[idx][1] <= y_range[1]:
            # _sc.show(crop=True, padding=20)
            # plt.show()
            if show_mask:
                mask_crop = _sc.get_mask_crop(padding=padding)
            img_crop = _sc.get_img_crop(padding=padding)

            if not (fix_dims is None):
                img_crop = crop_or_pad_img(img_crop, fix_dims=fix_dims)
                if show_mask:
                    mask_crop = crop_or_pad_img(mask_crop, fix_dims=fix_dims)
            _crops.append(img_crop)
            if show_mask:
                mask_crops.append(mask_crop)
            _crop_coords.append(embedding[idx])
        if len(_crops) >= max_crops:
            break

    if len(_crops) == 0:
        main_info(f"No single cell found in the embedding space")
        return

    if sort_by_x:
        main_info(f"Sorting crops by x")
        sort_indices = np.argsort(np.array(_crop_coords)[:, 0])
        _crops = [_crops[i] for i in sort_indices]
        if show_mask:
            mask_crops = [mask_crops[i] for i in sort_indices]
        _crop_coords = [_crop_coords[i] for i in sort_indices]

    # Visualize crops on one orw
    import matplotlib.pyplot as plt

    if show_mask:
        fig, axes = plt.subplots(2, len(_crops), figsize=(len(_crops) * 4, 10), dpi=dpi)
    else:
        fig, axes = plt.subplots(1, len(_crops), figsize=(len(_crops) * 4, 5), dpi=dpi)
        axes = [axes]

    for idx in range(len(_crops)):
        axes[0][idx].imshow(_crops[idx])
        axes[0][idx].axis("off")
        if show_mask:
            axes[1][idx].imshow(mask_crops[idx])
            axes[1][idx].axis("off")

    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=0, h_pad=0)
    return fig, axes
