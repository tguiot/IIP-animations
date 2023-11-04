import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve, gaussian_filter, median_filter, uniform_filter
from skimage import filters


def generate_gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel.

    Parameters:
        size (int): The size of the kernel. Must be odd.
        sigma (float): The standard deviation of the Gaussian function.

    Returns:
        numpy.ndarray: The Gaussian kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    offset = int(size // 2)
    x, y = np.mgrid[-offset : offset + 1, -offset : offset + 1]
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_kernel /= 2 * np.pi * sigma**2
    # Optionally, you could also normalize the kernel so that it sums to 1
    # gaussian_kernel /= gaussian_kernel.sum()

    return gaussian_kernel


# Average filtering
def average_filter(image, kernel_size=3):
    return uniform_filter(image, size=kernel_size, mode="nearest")


# Gaussian filtering
def gaussian_filtering(image, sigma=1):
    return gaussian_filter(image, sigma=sigma, mode="nearest")


# Median filtering
def median_filtering(image, kernel_size=3):
    return median_filter(image, size=kernel_size, mode="nearest")


def apply_custom_mask(image, mask):
    return convolve(image, mask, mode="nearest")


def apply_prewitt_filter(image):
    return filters.prewitt(image)


def apply_sobel_filter(image):
    return filters.sobel(image)
    # return ndimage.sobel(image, axis=0)


def apply_laplacian_filter(image, size=3):
    return filters.laplace(image, ksize=size)


def apply_sharpening_filter(image):
    laplace = filters.laplace(image, ksize=3)
    image = image.astype(np.float64) / 255
    print("type laplace:", laplace.dtype)
    print("type original:", image.dtype)
    # return cv2.addWeighted(image, 1.5, laplace, -0.5, 0)
    return 0.5 * image + 0.5 * laplace
