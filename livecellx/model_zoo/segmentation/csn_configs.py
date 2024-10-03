from functools import partial
import torch
from torchvision import transforms
from torchvision import transforms
from typing import Tuple

from livecellx.model_zoo.segmentation.custom_transforms import (
    CustomTransformV5,
    CustomTransformV7,
    CustomTransformEdtV8,
)


def gen_train_transform_v0(
    degrees: float, translation_range: Tuple[float, float], scale: Tuple[float, float]
) -> transforms.Compose:
    """Generate the training data transformation.

    Parameters
    ----------
    degrees : float
        The range of degrees to rotate the image.
    translation_range : Tuple[float, float]
        The range of translation in pixels.
    scale : Tuple[float, float]
        The range of scale factors.

    Returns
    -------
    transforms.Compose
        The composed transformation for training data.
    """

    train_transforms = transforms.Compose(
        [
            # transforms.Resize((412, 412)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=degrees, translate=translation_range, scale=scale),
            transforms.RandomCrop((412, 412), pad_if_needed=True),
        ]
    )
    return train_transforms


def gen_train_transform_v1(degrees=0, translation_range=None, scale=None) -> transforms.Compose:
    """Generate the training data transformation.

    Parameters
    ----------
    degrees : float
        The range of degrees to rotate the image.
    translation_range : Tuple[float, float]
        The range of translation in pixels.
    scale : Tuple[float, float]
        The range of scale factors.

    Returns
    -------
    transforms.Compose
        The composed transformation for training data.
    """

    train_transforms = transforms.Compose(
        [
            # transforms.Resize((412, 412)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=degrees, translate=translation_range, scale=scale),
            transforms.Resize((412, 412)),
        ]
    )
    return train_transforms


def gen_train_transform_v2(
    degrees: float, translation_range: Tuple[float, float], scale: Tuple[float, float]
) -> transforms.Compose:
    """Generate the training data transformation.

    Parameters
    ----------
    degrees : float
        The range of degrees to rotate the image.
    translation_range : Tuple[float, float]
        The range of translation in pixels.
    scale : Tuple[float, float]
        The range of scale factors.

    Returns
    -------
    transforms.Compose
        The composed transformation for training data.
    """

    train_transforms = transforms.Compose(
        [
            # transforms.Resize((412, 412)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=degrees, translate=translation_range, scale=scale),
        ]
    )
    return train_transforms


def gauss_noise_tensor(
    img,
    sigma=30.0,
):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    minus_or_plus = torch.randint(0, 2, (1,)).item()
    if minus_or_plus == 0:
        out = img + sigma * torch.randn_like(img)
    else:
        out = img - sigma * torch.randn_like(img)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out


def gen_train_transform_v3(
    degrees: float, translation_range: Tuple[float, float], scale: Tuple[float, float], gauss_sigma=30
) -> transforms.Compose:
    """Generate the training data transformation.

    Parameters
    ----------
    degrees : float
        The range of degrees to rotate the image.
    translation_range : Tuple[float, float]
        The range of translation in pixels.
    scale : Tuple[float, float]
        The range of scale factors.

    Returns
    -------
    transforms.Compose
        The composed transformation for training data.
    """

    train_transforms = transforms.Compose(
        [
            # transforms.Resize((412, 412)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=degrees, translate=translation_range, scale=scale),
            partial(gauss_noise_tensor, sigma=gauss_sigma),
            transforms.Resize((412, 412)),
        ]
    )
    return train_transforms


def gen_train_transform_v4(
    degrees: float, translation_range: Tuple[float, float], scale: Tuple[float, float], gauss_sigma=30
) -> transforms.Compose:
    """Generate the training data transformation.

    Parameters
    ----------
    degrees : float
        The range of degrees to rotate the image.
    translation_range : Tuple[float, float]
        The range of translation in pixels.
    scale : Tuple[float, float]
        The range of scale factors.

    Returns
    -------
    transforms.Compose
        The composed transformation for training data.
    """

    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=degrees, translate=translation_range, scale=scale, shear=10),
            partial(gauss_noise_tensor, sigma=gauss_sigma),
            transforms.Resize((412, 412)),
            transforms.Normalize([0.485], [0.229]),
        ]
    )
    return train_transforms


def gen_train_transform_v5(
    degrees: float, translation_range: Tuple[float, float] = None, scale: Tuple[float, float] = None, gauss_sigma=30
) -> CustomTransformV5:
    """Generate the training data transformation.

    Parameters
    ----------
    degrees : float
        The range of degrees to rotate the image.
    translation_range : Tuple[float, float]
        The range of translation in pixels.
    scale : Tuple[float, float]
        The range of scale factors.

    Returns
    -------
    transforms.Compose
        The composed transformation for training data.
    """

    train_transforms = CustomTransformV5(degrees=degrees, translation_range=translation_range, scale=scale)
    return train_transforms


def gen_train_transform_v6(
    degrees: float, translation_range: Tuple[float, float] = None, scale: Tuple[float, float] = None, gauss_sigma=30
) -> transforms.Compose:
    """Generate the training data transformation.

    Parameters
    ----------
    degrees : float
        The range of degrees to rotate the image.
    translation_range : Tuple[float, float]
        The range of translation in pixels.
    scale : Tuple[float, float]
        The range of scale factors.

    Returns
    -------
    transforms.Compose
        The composed transformation for training data.
    """

    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=degrees, translate=translation_range, scale=scale, shear=10),
            transforms.Resize((256, 256)),
        ]
    )
    return train_transforms


def gen_train_transform_v7(
    degrees: float, translation_range: Tuple[float, float] = None, scale: Tuple[float, float] = None, gauss_sigma=30
) -> CustomTransformV5:
    """Generate the training data transformation.

    Parameters
    ----------
    degrees : float
        The range of degrees to rotate the image.
    translation_range : Tuple[float, float]
        The range of translation in pixels.
    scale : Tuple[float, float]
        The range of scale factors.

    Returns
    -------
    transforms.Compose
        The composed transformation for training data.
    """

    train_transforms = CustomTransformV7(degrees=degrees, translation_range=translation_range, scale=scale)
    return train_transforms


def gen_train_transform_edt_v8(**kwargs) -> CustomTransformEdtV8:
    """Generate the training data transformation.

    Parameters
    ----------
    degrees : float
        The range of degrees to rotate the image.
    translation_range : Tuple[float, float]
        The range of translation in pixels.
    scale : Tuple[float, float]
        The range of scale factors.

    Returns
    -------
    transforms.Compose
        The composed transformation for training data.
    """

    train_transforms = CustomTransformEdtV8(**kwargs)
    return train_transforms
