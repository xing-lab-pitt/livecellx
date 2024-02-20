from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
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
    scs,
    padding=40,
    x_range=(float("-inf"), float("inf")),
    y_range=(float("-inf"), float("inf")),
    title="Single cell in Embedding Space",
    max_crops=10,
    fix_dims=None,
    randomly_select=True,
):
    _crops = []
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
            img_crop = _sc.get_img_crop(padding=padding)
            if not (fix_dims is None):
                img_crop = crop_or_pad_img(img_crop, fix_dims=fix_dims)
            _crops.append(img_crop)
        if len(_crops) >= max_crops:
            break

    if len(_crops) == 0:
        main_info(f"No single cell found in the embedding space")
        return
    # Visualize crops on one orw
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(_crops), figsize=(len(_crops) * 4, 5), dpi=300)
    for idx, ax in enumerate(axes):
        ax.imshow(_crops[idx])
        ax.axis("off")
    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=0, h_pad=0)
    return fig, axes
