import torch
from torchvision import transforms
from typing import Tuple

from livecellx.preprocess.utils import normalize_edt


class CustomTransformV5:
    def __init__(
        self,
        degrees: float,
        translation_range: Tuple[float, float] = None,
        scale: Tuple[float, float] = None,
    ):
        # Common transformations that should be applied to both images and masks
        self.common_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=degrees, translate=translation_range, scale=scale, shear=10),
                transforms.Resize((256, 256)),
            ]
        )
        # Image-specific transformations that should not be applied to masks
        self.image_transforms = transforms.Compose(
            [
                transforms.GaussianBlur(kernel_size=3, sigma=30),
                transforms.Normalize([127], [30]),  # Adjust channel numbers according to your images
            ]
        )

    def apply_common_transforms(self, tensor):
        # Assuming tensor is a PyTorch tensor, you might need to convert it to PIL Image first
        # Depending on your specific setup, conversion between PIL Images and tensors may be required
        tensor = self.common_transforms(tensor)
        return tensor

    def apply_image_transforms(self, image):
        # Apply transformations specific to images
        # Adjustments might be necessary depending on whether your data is in PIL Image or tensor format
        image = self.image_transforms(image)
        return image

    def __call__(self, concat_img):
        # Apply common transformations
        concat_img = self.apply_common_transforms(concat_img)

        # Apply image-specific transformations
        # Assuming the first two images in concat_img are the ones needing image-specific transformations
        # for i in range(2): # Adjust this range based on how many images you have that need these transformations
        concat_img[:2] = self.apply_image_transforms(concat_img[:2])

        return concat_img


class CustomTransformV7:
    def __init__(
        self,
        degrees: float,
        translation_range: Tuple[float, float] = None,
        scale: Tuple[float, float] = None,
    ):
        self.common_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=degrees, translate=translation_range, scale=scale, shear=10),
            ]
        )
        # Common transformations that should be applied to both images and masks
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )
        # Image-specific transformations that should not be applied to masks
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.GaussianBlur(kernel_size=3, sigma=30),
                transforms.Normalize([127], [30]),  # Adjust channel numbers according to your images
            ]
        )

    def apply_mask_transforms(self, tensor):
        # Assuming tensor is a PyTorch tensor, you might need to convert it to PIL Image first
        # Depending on your specific setup, conversion between PIL Images and tensors may be required
        tensor = self.mask_transforms(tensor)
        return tensor

    def apply_image_transforms(self, image):
        # Apply transformations specific to images
        # Adjustments might be necessary depending on whether your data is in PIL Image or tensor format
        image = self.image_transforms(image)
        return image

    def __call__(self, concat_img):
        # Apply common transformations
        concat_img = self.common_transforms(concat_img)

        # Apply mask transformations
        transformed_mask = self.apply_mask_transforms(concat_img[2:])

        # Apply image-specific transformations
        # Assuming the first two images in concat_img are the ones needing image-specific transformations
        # for i in range(2): # Adjust this range based on how many images you have that need these transformations
        transformed_image = self.apply_image_transforms(concat_img[:2])

        concat_img = torch.cat((transformed_image, transformed_mask), dim=0)
        return concat_img


class CustomTransformEdtV8:
    def __init__(
        self,
        degrees: float = 0,
        translation_range: Tuple[float, float] = None,
        scale: Tuple[float, float] = None,
        shear=0,
        flip_p=0,
        gaussian_blur_kernel_size=3,
        gaussian_blur_sigma=30,
        use_gaussian_blur=True,
    ):
        self.common_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=flip_p),
                transforms.RandomVerticalFlip(p=flip_p),
                transforms.RandomAffine(
                    degrees=degrees,
                    translate=translation_range,
                    scale=scale,
                    shear=shear,
                ),
            ]
        )
        # Common transformations that should be applied to both images and masks
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )
        # Image-specific transformations that should not be applied to masks
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
                (
                    transforms.GaussianBlur(kernel_size=gaussian_blur_kernel_size, sigma=gaussian_blur_sigma)
                    if use_gaussian_blur
                    else transforms.Lambda(lambda x: x)
                ),
                transforms.Normalize([127], [30]),  # Adjust channel numbers according to your images
            ]
        )
        self.degree = degrees
        self.translation_range = translation_range
        self.scale = scale

    def apply_mask_transforms(self, tensor):
        # Assuming tensor is a PyTorch tensor, you might need to convert it to PIL Image first
        # Depending on your specific setup, conversion between PIL Images and tensors may be required
        tensor = self.mask_transforms(tensor)
        return tensor

    def apply_image_transforms(self, image):
        # Apply transformations specific to images
        # Adjustments might be necessary depending on whether your data is in PIL Image or tensor format
        image = self.image_transforms(image)
        return image

    def __call__(self, concat_img):
        # Apply common transformations
        concat_img = self.common_transforms(concat_img)

        # Apply mask transformations
        transformed_mask = self.apply_mask_transforms(concat_img[2:])

        # Apply image-specific transformations
        # Assuming the first two images in concat_img are the ones needing image-specific transformations
        # for i in range(2): # Adjust this range based on how many images you have that need these transformations
        transformed_image = self.apply_image_transforms(concat_img[:2])

        concat_img = torch.cat((transformed_image, transformed_mask), dim=0)
        # Normalize EDT channel: [1, max_edt]
        max_edt = 4
        edt_img = concat_img[7]
        max_val = edt_img.max()
        factor = max_val / max_edt
        edt_pos_mask = edt_img >= 1
        edt_img[edt_pos_mask] = edt_img[edt_pos_mask] / factor + 1
        concat_img[7] = edt_img

        return concat_img


class CustomTransformEdtV9:
    def __init__(
        self,
        degrees: float = 0,
        translation_range: Tuple[float, float] = None,
        scale: Tuple[float, float] = None,
        shear=0.0,
        flip_p=0.0,
        gaussian_blur_kernel_size=3,
        gaussian_blur_sigma=(0, 30),
        use_gaussian_blur=False,
    ):
        self.common_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=flip_p),
                transforms.RandomVerticalFlip(p=flip_p),
                transforms.RandomAffine(
                    degrees=degrees,
                    translate=translation_range,
                    scale=scale,
                    shear=shear,
                ),
            ]
        )
        # Common transformations that should be applied to both images and masks
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )
        # Image-specific transformations that should not be applied to masks
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
                (
                    transforms.GaussianBlur(kernel_size=gaussian_blur_kernel_size, sigma=gaussian_blur_sigma)
                    if use_gaussian_blur
                    else transforms.Lambda(lambda x: x)
                ),
            ]
        )
        self.degree = degrees
        self.translation_range = translation_range
        self.scale = scale

    def apply_mask_transforms(self, tensor):
        # Assuming tensor is a PyTorch tensor, you might need to convert it to PIL Image first
        # Depending on your specific setup, conversion between PIL Images and tensors may be required
        tensor = self.mask_transforms(tensor)
        return tensor

    def apply_image_transforms(self, image):
        # Apply transformations specific to images
        # Adjustments might be necessary depending on whether your data is in PIL Image or tensor format
        image = self.image_transforms(image)
        return image

    def __call__(self, concat_img):
        # Apply common transformations
        concat_img = self.common_transforms(concat_img)

        # Apply mask transformations
        transformed_mask = self.apply_mask_transforms(concat_img[2:])

        # Apply image-specific transformations
        # Assuming the first two images in concat_img are the ones needing image-specific transformations
        # for i in range(2): # Adjust this range based on how many images you have that need these transformations
        transformed_image = self.apply_image_transforms(concat_img[:2])

        concat_img = torch.cat((transformed_image, transformed_mask), dim=0)
        # Normalize EDT channel: [1, max_edt]
        edt_img = normalize_edt(concat_img[7], edt_max=5)
        concat_img[7] = edt_img

        return concat_img
