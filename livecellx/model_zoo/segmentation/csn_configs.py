from torchvision import transforms
from torchvision import transforms
from typing import Tuple


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


def gen_train_transform_v1(
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
