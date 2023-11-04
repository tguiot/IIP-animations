import numpy as np
from matplotlib import pyplot as plt

from filtering.filters import (
    apply_laplacian_filter,
    apply_prewitt_filter,
    apply_sharpening_filter,
    apply_sobel_filter,
    average_filter,
    gaussian_filtering,
    median_filtering,
)


def plot3(image0, image1, image2, desc1, desc2):
    # Display images
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image0, cmap="gray")
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.title(desc1)
    plt.imshow(image1, cmap="gray")
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 3, 3)
    plt.title(desc2)
    plt.imshow(image2, cmap="gray")
    plt.xticks([]), plt.yticks([])

    plt.show()


def plot2(image0, image1, desc1):
    # Display images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(image0, cmap="gray")
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.title(desc1)
    plt.imshow(image1, cmap="gray")
    plt.xticks([]), plt.yticks([])

    plt.show()


def plot_average(frame):
    # Apply filters
    avg_filtered_frame_3 = average_filter(frame)
    avg_filtered_frame_5 = average_filter(frame, 5)
    gaussian_filtered_frame = gaussian_filtering(frame)
    median_filtered_frame = median_filtering(frame)
    plot3(frame, avg_filtered_frame_3, avg_filtered_frame_5, "Average 3x3", "Average 5x5")


def plot_gaussian(image):
    # Apply filters
    gaussian_1 = gaussian_filtering(image, 1)
    gaussian_2 = gaussian_filtering(image, 2)
    plot3(image, gaussian_1, gaussian_2, "Gaussian, sigma=1", "Gaussian, sigma=2")


def plot_median(image):
    # Apply filters
    median_3 = median_filtering(image, 3)
    median_5 = median_filtering(image, 5)
    plot3(image, median_3, median_5, "Median filter 3x3", "Median filter 5x5")


def plot_prewitt(image):
    prewitted = apply_prewitt_filter(image)
    plot2(image, prewitted, "Prewitt result")


def plot_prewitt_sobel(image):
    prewitt = apply_prewitt_filter(image)
    sobel = apply_sobel_filter(image)
    plot3(image, prewitt, sobel, "Prewitt", "Sobel")


def plot_laplacian(image):
    laplace = apply_laplacian_filter(image)
    plot2(image, laplace, "Laplacien")


def plot_sharpening(image):
    sharpened = apply_sharpening_filter(image)
    plot2(image, sharpened, "Sharpened")
