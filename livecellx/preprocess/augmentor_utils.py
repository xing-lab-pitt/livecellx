import Augmentor
import numpy as np


def augment_images_by_augmentor(images: np.ndarray, masks: np.ndarray, crop_image_size: int, sampling_amount: int):
    """Augments images and masks using Augmentor library.

    Parameters
    ----------
    images : np.ndarray
        N x w x h x channel
    masks : np.ndarray
        N x w x h x channel
    crop_image_size :
        size of the cropped image (outputs)
    sampling_amount :
        amount of images to be sampled from Augmentor

    Returns
    -------
    _type_
        _description_
    """
    n_ch = images[0].shape[2]
    images = [np.moveaxis(images[i], 2, 0) for i in range(len(images))]
    masks = [np.moveaxis(masks[i], 2, 0) for i in range(len(masks))]

    combined_imgs = [np.concatenate([images[i], masks[i]], axis=0) for i in range(len(images))]

    p = Augmentor.DataPipeline(combined_imgs)
    p.flip_left_right(0.5)
    p.flip_top_bottom(0.5)
    p.rotate(1, max_left_rotation=10, max_right_rotation=10)
    # p.zoom_random(1, percentage_area=0.5)
    p.zoom(probability=0.9, min_factor=0.1, max_factor=3)
    p.crop_by_size(1, crop_image_size, crop_image_size, centre=False)
    augmented_combined_images = p.sample(sampling_amount)
    augmented_combined_images = np.array(augmented_combined_images)

    augmented_combined_images = np.moveaxis(augmented_combined_images, 1, 3)
    augmented_images, masks = augmented_combined_images[..., :n_ch], augmented_combined_images[..., n_ch:]

    # augmented_combined_images = [np.moveaxis(augmented_combined_images[i], 0, 2) for i in range(len(augmented_combined_images))]
    # augmented_images  = [augmented_combined_images[i][..., :n_ch] \
    #                      for i in range(len(augmented_combined_images))]
    # masks = [augmented_combined_images[i][..., n_ch:] \
    #          for i in range(len(augmented_combined_images))]
    return augmented_images, masks
