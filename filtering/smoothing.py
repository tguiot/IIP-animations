import cv2
import imageio

from filtering.filters import gaussian_filtering
from filtering.plotters import plot_laplacian, plot_prewitt, plot_prewitt_sobel, plot_sharpening

if __name__ == "__main__":
    # Load the GIF file
    reader = imageio.get_reader("ct2.gif")
    frames = list(reader)

    # Get the first frame (or choose any frame)
    frame = frames[0]
    image = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    print("Image dimensions:", image.shape)

    # plot_average(frame)
    # plot_gaussian(frame)
    # plot_gaussian_custom(frame)
    # plot_median(frame)
    image = gaussian_filtering(image)
    # plot_prewitt_sobel(image)

    # plot_laplacian(image)
    plot_sharpening(image)
