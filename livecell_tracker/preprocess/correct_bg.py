import numpy as np
import scipy.interpolate as interp
from skimage import exposure, filters, morphology


def correct_background_median_gamma(image: np.ndarray, disk_size: int = 15) -> np.ndarray:
    """Corrects the background of an image using local median filter and gamma correction.

    Args:
        image (np.ndarray): Input image as a numpy array.
        disk_size (int): Size of the disk structuring element used for median filter.

    Returns:
        np.ndarray: Background-corrected image as a numpy array.

    """
    # Apply local median filter to remove noise
    selem = morphology.disk(disk_size)
    image_median = filters.rank.median(image, selem=selem)

    # Subtract background
    image_bg = exposure.adjust_gamma(image_median, gamma=2)
    image_corrected = exposure.rescale_intensity(image - image_bg, out_range=(0, 1))

    return image_corrected


def correct_background_bisplrep(image: np.ndarray, kx: int = 3, ky: int = 3, s: float = 0) -> np.ndarray:
    """Corrects the background of an image using 2D B-spline.

    Args:
        image (np.ndarray): Input image as a numpy array.
        kx (int): Degree of the B-spline in the x-direction.
        ky (int): Degree of the B-spline in the y-direction.
        s (float): Smoothing factor for the B-spline.

    Returns:
        np.ndarray: Background-corrected image as a numpy array.

    """
    # Define grid of coordinates
    ygrid, xgrid = np.indices(image.shape)

    # Fit B-spline to image
    tck = interp.bisplrep(xgrid.ravel(), ygrid.ravel(), image.ravel(), kx=kx, ky=ky, s=s)

    # Evaluate B-spline on grid
    image_bg = interp.bisplev(xgrid.ravel(), ygrid.ravel(), tck)
    image_bg = image_bg.reshape(image.shape)

    # Subtract background
    image_corrected = image - image_bg
    image_corrected = np.clip(image_corrected, 0, 1)

    return image_corrected
