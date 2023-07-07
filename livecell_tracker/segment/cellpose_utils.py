import numpy as np
from livecell_tracker.preprocess.utils import normalize_img_to_uint8


def segment_single_image_by_cellpose(image, model, channels=[[0, 0]], diameter=150) -> np.array:
    result_tuple = model.eval([image], diameter=diameter, channels=channels)
    masks = result_tuple[0]
    return np.array(masks[0])


def segment_single_images_by_cellpose(images, model, channels=[[0, 0]], diameter=150):
    masks, flows, styles, diams = model.eval(images, diameter=diameter, channels=channels)
    return masks
