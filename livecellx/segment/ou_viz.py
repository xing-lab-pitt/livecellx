import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import skimage
from skimage.morphology import local_maxima, h_maxima
from skimage.measure import regionprops, label

from livecellx.core.single_cell import SingleCellStatic
from livecellx.segment.ou_utils import create_ou_input_from_sc
from livecellx.preprocess.utils import normalize_img_to_uint8, enhance_contrast


def viz_ou_sc_outputs(
    sc: SingleCellStatic,
    model,
    transforms,
    padding_pixels: int = 0,
    dtype=float,
    remove_bg=True,
    one_object=True,
    scale=0,
    out_threshold=1,
    save_path=None,
    show=True,
):
    ou_input = create_ou_input_from_sc(
        sc, padding_pixels=padding_pixels, dtype=dtype, remove_bg=remove_bg, one_object=one_object, scale=scale
    )
    viz_ou_outputs(
        ou_input,
        sc.get_sc_mask(padding=padding_pixels, dtype=int),
        model,
        transforms,
        out_threshold=out_threshold,
        original_img=sc.get_img_crop(padding=padding_pixels),
        save_path=save_path,
        show=show,
    )


def viz_ou_outputs(
    ou_input, original_mask, model, input_transforms, out_threshold, show=True, original_img=None, save_path=None
):
    original_shape = ou_input.shape
    original_ou_input = ou_input.copy()
    ou_input = input_transforms(torch.tensor([ou_input]))
    ou_input = torch.stack([ou_input, ou_input, ou_input], dim=1)
    ou_input = ou_input.float().cuda()
    output = model(ou_input)

    # transform (resize) output to original shape
    back_transforms = transforms.Compose(
        [
            transforms.Resize(size=(original_shape[0], original_shape[1])),
        ]
    )
    output = back_transforms(output)

    # perform watershed on output
    marker_method = "hmax"
    h_threshold = 20

    # marker_method = "local"
    # peak_distance = 50
    markers = None
    edt_distance = output.cpu().detach().numpy()[0, 0]

    if markers is None and marker_method == "hmax":
        # local_hmax = h_maxima(raw_crop, h_threshold)
        local_hmax = h_maxima(edt_distance, h_threshold)
        markers = skimage.measure.label(local_hmax, connectivity=1)
    elif markers is None and marker_method == "local":
        # use local peak as default markers
        coords = peak_local_max(edt_distance, min_distance=peak_distance, footprint=np.ones((3, 3)))
        mask = np.zeros(edt_distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)

    # labels = watershed(edt_distance, markers, mask=contour_mask)
    if show:
        fig, axes = plt.subplots(1, 2, figsize=(8, 2))
        axes[0].imshow(markers)
        axes[0].set_title("markers")
        axes[1].imshow(edt_distance)
        axes[1].set_title("edt_distance")
        plt.show()

    # watershed_mask = watershed(-edt_distance, markers, mask=original_mask)
    watershed_mask = watershed(-edt_distance, markers, mask=edt_distance > out_threshold)

    # visualize the input and all 3 output channels
    if show or (save_path is not None):
        if original_img is not None:
            num_figures = 8
        else:
            num_figures = 7
        fig, axes = plt.subplots(1, num_figures, figsize=(15, 5))
        axes[0].imshow(original_ou_input)
        axes[0].set_title("input")
        axes[1].imshow(output[0, 0].cpu().detach().numpy())
        axes[1].set_title("output c0")
        axes[2].imshow(output[0, 1].cpu().detach().numpy())
        axes[2].set_title("output c1")
        axes[3].imshow(output[0, 2].cpu().detach().numpy())
        axes[3].set_title("output c2")
        axes[4].imshow(original_mask)
        axes[4].set_title("original mask")
        axes[5].imshow(output[0, 0].cpu().detach().numpy() > out_threshold)
        axes[5].set_title("output c0 > 1")
        axes[6].imshow(watershed_mask)
        axes[6].set_title("watershed mask")
        if original_img is not None:
            axes[7].imshow(enhance_contrast(normalize_img_to_uint8(original_img)))
            axes[7].set_title("original img")
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)

    return output, watershed_mask
