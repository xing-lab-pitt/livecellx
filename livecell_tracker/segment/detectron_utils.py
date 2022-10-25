import os

import matplotlib.pyplot as plt
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from skimage import measure
from livecell_tracker.core import SingleCellTrajectory, SingleCellStatic
from pathlib import Path
from livecell_tracker.core.datasets import LiveCellImageDataset
from PIL import Image, ImageSequence
from tqdm import tqdm
import json
from livecell_tracker.preprocess.utils import normalize_img_by_zscore


def gen_cfg(
    model_path=None,
    output_dir="./detectron_training_output",
    train_dataset_name="deepfashion_train",
    test_dataset_name="deepfashion_val",
    config_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    checkpoint_url="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)  # Let training initialize from model zoo

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13

    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = output_dir
    if model_path is not None:
        cfg.MODEL.WEIGHTS = model_path  # os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    else:
        cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR) / "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set the testing threshold for this model
    cfg.DATASETS.TEST = (test_dataset_name,)
    return cfg


def detectron_visualize_img(img, cfg, detectron_outputs):
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(detectron_outputs["instances"].to("cpu"))
    axis_img = v.get_image()[:, :, ::-1]
    return axis_img


def convert_detectron_instances_to_label_masks(instance_pred_masks):
    res_mask = np.zeros(instance_pred_masks.shape[1:])
    for idx in range(instance_pred_masks.shape[0]):
        res_mask[instance_pred_masks[idx, :, :]] = idx + 1
    return res_mask


def convert_detectron_instance_pred_masks_to_binary_masks(instance_pred_masks):
    label_mask = convert_detectron_instances_to_label_masks(instance_pred_masks)
    label_mask[label_mask > 0] = 1
    return label_mask


def segment_by_detectron(img, detectron_predictor):
    outputs = detectron_predictor(img)
    return outputs


def segment_detectron_wrapper(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    results = segment_by_detectron(normalize_img_by_zscore(img))
    instances = results["instances"].to("cpu").pred_masks.numpy()
    mask = convert_detectron_instance_pred_masks_to_binary_masks(instances)
    return mask


def segment_single_img_by_detectron_wrapper(img, predictor, return_detectron_results=True):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    results = segment_by_detectron(img, predictor)
    instance_pred_masks = results["instances"].to("cpu").pred_masks.numpy()
    if return_detectron_results:
        return instance_pred_masks, results
    return instance_pred_masks


def segment_images_by_detectron(
    img_dataset: LiveCellImageDataset,
    out_dir: Path,
    cfg=None,
    return_path_to_contours=True,
):
    """segment images by detectron2

    Parameters
    ----------
    imgs : LiveCellImageDataset
        _description_
    cfg : _type_
        _description_
    out_dir : Path
        _description_

    Returns
    -------
    _type_
        if return path to contours is true, return a dictionary of image path to contours
        else return a list of single cell objects
    """
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    predictor = DefaultPredictor(cfg)
    segmentation_results = {}
    all_single_cells = []
    for timeframe in tqdm(range(len(img_dataset))):
        img_path = img_dataset.get_img_path(timeframe)
        img = img_dataset[timeframe]
        original_img_filename = os.path.basename(img_path).split(".")[0]
        output_filename = original_img_filename + ".png"  # change extension to PNG

        (
            instance_pred_masks,
            predictor_results,
        ) = segment_single_img_by_detectron_wrapper(img, predictor=predictor, return_detectron_results=True)
        if instance_pred_masks.max() >= 2 ** 8:
            # TODO: logger
            print("[WARNING] more than 256 instances predicted, potential overflow")

        # save binary mask
        def _save_binary_mask():
            binary_mask = convert_detectron_instance_pred_masks_to_binary_masks(instance_pred_masks)
            # convert mask to 8-bit binary mask
            binary_mask = binary_mask.astype(np.uint8)
            binary_mask_img = Image.fromarray(binary_mask)
            binary_mask_img.save(out_dir / output_filename)
            del binary_mask, binary_mask_img

        def _save_overlay_img():
            # save overlayed image
            overlay_output_filename = "overlay_" + original_img_filename + ".png"  # change extension to PNG
            # overlayed_img = overlay(img, mask, mask_channel_rgb_val=100, img_channel_rgb_val_factor=2)
            overlayed_arr = detectron_visualize_img(img[:, :, np.newaxis], cfg, predictor_results)
            overlayed_img = Image.fromarray(overlayed_arr)
            overlayed_img.save(out_dir / overlay_output_filename)
            del overlayed_img, overlayed_arr

        def _save_instance_masks():
            # save predicted instance masks
            pred_binary_masks = predictor_results["instances"].to("cpu").pred_masks.numpy()
            for idx in range(pred_binary_masks.shape[0]):
                pred_binary_mask = pred_binary_masks[idx, :, :]
                pred_binary_mask_img = Image.fromarray(pred_binary_mask)
                pred_binary_mask_img.save(out_dir / f"{original_img_filename}_instance_{idx}.png")
                del pred_binary_mask, pred_binary_mask_img
            del pred_binary_masks

        _save_binary_mask()
        # _save_instance_masks()
        _save_overlay_img()
        # generate contours and save to json
        contours = get_contours_from_pred_masks(instance_pred_masks)
        single_cells = [
            SingleCellStatic(timeframe=timeframe, contour=contour, img_dataset=img_dataset) for contour in contours
        ]
        all_single_cells.extend(single_cells)
        assert original_img_filename not in segmentation_results, "duplicate image filename?"
        segmentation_results[img_path] = {}
        segmentation_results[img_path]["contours"] = contours
    if return_path_to_contours:
        return segmentation_results
    else:
        return single_cells


def get_contours_from_pred_masks(instance_pred_masks):
    contours = []
    for instance_mask in instance_pred_masks:
        tmp_contours = measure.find_contours(
            instance_mask, level=0.5, fully_connected="low", positive_orientation="low"
        )
        if len(tmp_contours) != 1:
            print("[WARN] more than 1 contour found in the instance mask")
        # convert to list for saving into json
        contours.extend([[list(coords) for coords in coord_arr] for coord_arr in tmp_contours])
    return contours
