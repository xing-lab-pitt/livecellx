import random
from matplotlib import pyplot as plt
from mmcv.transforms import BaseTransform

# from mmdet.registry import TRANSFORMS
# from mmaction.datasets.builder import TRANSFORMS
from mmaction.registry import TRANSFORMS
import numpy as np
from skimage.measure import regionprops

from livecellx.preprocess.utils import dilate_or_erode_mask


@TRANSFORMS.register_module()
class RandomMaskCentroid:
    """LivecellAction customized augmentation
    For channel blue and green in a video, randomly mask out all values by thesholds lb and ub.
    Purpose: we would like to control the amount of prior segmentation information fed to the models.
    """

    def __init__(self, lb=2, ub=50):
        self.lb = lb
        self.ub = ub

    def check_channels(self, img):
        """blue and green channel should have the same values"""
        print("sum of red channel: ", img[:, :, 0].sum(), "nonzero: ", (img[:, :, 0] != 0).sum())
        print("sum of green channel: ", img[:, :, 1].sum(), "nonzero: ", (img[:, :, 1] != 0).sum())
        print("sum of blue channel: ", img[:, :, 2].sum(), "nonzero: ", (img[:, :, 2] != 0).sum())
        # plt.imshow(img[:,:,0])
        # plt.savefig("tmp/debug_red.png")
        # plt.imshow(img[:,:,1])
        # plt.savefig("tmp/debug_green.png")
        # plt.imshow(img[:,:,2])
        # plt.savefig("tmp/debug_blue.png")
        assert (img[:, :, 0] == img[:, :, 1]).all(), "blue and green channel should have the same values"

    def __call__(self, results, exact_match_rg_channel=False):

        imgs = results["imgs"]
        img_h, img_w = imgs[0].shape[:2]

        # Because of lossy compression, some masks can have random dots...
        scale_factor = -0.01

        # Randomly [self.lb, self.ub) mask out the blue and green channels]
        lb = max(1, self.lb)
        if self.ub is None:
            ub = img_w
        else:
            ub = min(img_w, self.ub)

        centroid_box_width = np.random.randint(lb, ub)
        for i, img in enumerate(imgs):
            new_img = img.copy()
            corrected_mask = dilate_or_erode_mask(img[:, :, 0], scale_factor=scale_factor)
            properties = regionprops(corrected_mask.astype(int))
            # Sort by area
            properties.sort(key=lambda x: x.area, reverse=True)
            props = properties[0]
            centroid = props.centroid
            centroid_box = np.array(
                [
                    centroid[0] - centroid_box_width / 2,
                    centroid[1] - centroid_box_width / 2,
                    centroid[0] + centroid_box_width / 2,
                    centroid[1] + centroid_box_width / 2,
                ]
            )
            # Clip to image size
            centroid_box[0] = max(0, centroid_box[0])
            centroid_box[1] = max(0, centroid_box[1])
            centroid_box[2] = min(img_w, centroid_box[2])
            centroid_box[3] = min(img_h, centroid_box[3])

            # Draw centroid box via corrected_mask mass weight
            new_mask = np.zeros(corrected_mask.shape)
            new_mask[int(centroid_box[1]) : int(centroid_box[3]), int(centroid_box[0]) : int(centroid_box[2])] = 1
            new_mask = new_mask.astype(int)

            new_img[:, :, 0] = new_mask
            new_img[:, :, 1] = new_mask

            # imgs[i] = dilate_or_erode_mask(img, scale_factor=0.99)
            imgs[i] = new_img.astype(np.uint8)

        # If require exact match for mask channels
        if exact_match_rg_channel:
            for img in imgs:
                self.check_channels(img)

        return results
