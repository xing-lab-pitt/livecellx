import numpy as np
from detectron_utils import convert_detectron_instances_to_binary_masks

from livecell_tracker.preprocess.utils import normalize_img_by_zscore


def segment_single_image_by_cellpose(image, model, channels=[[0, 0]], diameter=150):
    result_tuple = model.eval([image], diameter=diameter, channels=channels)
    masks = result_tuple[0]
    return masks[0]


def segment_single_images_by_cellpose(images, model, channels=[[0, 0]], diameter=150):
    masks, flows, styles, diams = model.eval(images, diameter=diameter, channels=channels)
    return masks


def segment_by_detectron(img, detectron_predictor):
    outputs = detectron_predictor(img)
    return outputs


def segment_cellpose_wrapper(img, model):
    return segment_single_image_by_cellpose(img, model, diameter=173)


def segment_detectron_wrapper(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    results = segment_by_detectron(normalize_img_by_zscore(img))
    instances = results["instances"].to("cpu").pred_masks.numpy()
    mask = convert_detectron_instances_to_binary_masks(instances)
    return mask


def segment_raw_img_by_detectron_wrapper(img, return_detectron_results=False):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    results = segment_by_detectron(img)
    instance_pred_masks = results["instances"].to("cpu").pred_masks.numpy()
    mask = convert_detectron_instances_to_binary_masks(instance_pred_masks)
    if return_detectron_results:
        return mask, results
    return mask
