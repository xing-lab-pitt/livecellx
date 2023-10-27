import numpy as np
import scipy
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
    footprint = morphology.disk(disk_size)
    image_median = filters.rank.median(image, footprint=footprint)

    # Subtract background
    image_bg = exposure.adjust_gamma(image_median, gamma=2)
    image_corrected = exposure.rescale_intensity(image - image_bg, out_range=(0, 1))

    return image_corrected


def correct_background_bisplrep(
    image: np.ndarray, kx: int = 3, ky: int = 3, s: float = 1e20, sample_step: int = 1
) -> np.ndarray:
    """Corrects the background of an image using 2D B-spline.

    Args:
        image (np.ndarray): Input image as a numpy array.
        kx (int): Degree of the B-spline in the x-direction.
        ky (int): Degree of the B-spline in the y-direction.
        s (float): Smoothing factor for the B-spline.
        sample_step (int): Step size for subsampling the image.

    Returns:
        np.ndarray: Background-corrected image as a numpy array.

    """
    # Define grid of coordinates
    ygrid, xgrid = np.indices(image.shape)

    # Subsample the image
    ygrid = ygrid[::sample_step, ::sample_step]
    xgrid = xgrid[::sample_step, ::sample_step]
    image_sub = image[::sample_step, ::sample_step]

    # Fit B-spline to subsampled image
    tck = interp.bisplrep(xgrid, ygrid, image_sub, kx=kx, ky=ky, s=s)

    # Evaluate B-spline on grid
    tck = scipy.interpolate.bisplrep(xgrid, ygrid, image_sub, s=s)  # xgrid and y grid are auto raveled
    nx, ny = image.shape[0], image.shape[1]
    lx = np.linspace(0, nx, nx)
    ly = np.linspace(0, ny, ny)
    image_corrected = scipy.interpolate.bisplev(lx, ly, tck)
    image_corrected = image - image_corrected
    image_corrected[image_corrected < 0] = 0

    return image_corrected


def correct_background_polyfit(image: np.ndarray, degree: int = 2) -> np.ndarray:
    """Corrects the background of an image using polynomial fitting and least squares regression.

    Args:
        image (np.ndarray): Input image as a numpy array.
        degree (int): Degree of the polynomial fit.

    Returns:
        np.ndarray: Background-corrected image as a numpy array.

    """
    # Define grid of coordinates
    ygrid, xgrid = np.indices(image.shape)

    # Flatten grid coordinates and image intensity values
    x = xgrid.ravel()
    y = ygrid.ravel()
    z = image.ravel()

    # Create polynomial matrix
    poly_matrix = np.column_stack([np.ones_like(x), x, y])
    for i in range(2, degree + 1):
        for j in range(i + 1):
            poly_matrix = np.column_stack([poly_matrix, x ** (i - j) * y**j])

    # Fit polynomial using least squares regression
    coeffs, _, _, _ = np.linalg.lstsq(poly_matrix, z, rcond=None)

    # Evaluate polynomial on grid
    image_bg = np.dot(poly_matrix, coeffs).reshape(image.shape)

    # Subtract background
    image_corrected = image - image_bg
    # make all image_corrected positive
    image_corrected += np.abs(np.min(image_corrected))

    return image_corrected
