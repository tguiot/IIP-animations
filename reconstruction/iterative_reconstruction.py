import matplotlib.pyplot as plt
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import iradon, radon, rescale

plt.ion()

fig, axs = plt.subplots(2, 3, figsize=(20, 10))

# Definition of the true object
activity_level = 0.1
true_object = shepp_logan_phantom()
true_object = rescale(activity_level * true_object, 0.5)
axs[0, 0].imshow(true_object, cmap="Greys_r")
axs[0, 0].set_title("Object")

# Generate simulated sinogram data
azi_angles = np.linspace(0.0, 180.0, 180, endpoint=False)
sinogram = radon(true_object, azi_angles, circle=False)
axs[0, 1].imshow(sinogram.T, cmap="Greys_r")
axs[0, 1].set_title("Sinogram")

# Define image estimation at iteration zero
mlem_rec = np.ones(true_object.shape)

# Define the sensitivity image
sino_ones = np.ones(sinogram.shape)
sens_image = iradon(sino_ones, azi_angles, circle=False, filter_name=None)

# Iterate
for iter in range(50):
    fp = radon(mlem_rec, azi_angles, circle=False)  # Forward projection
    # division (with safety offset in case of zero values)
    ratio = sinogram / (fp + 0.000001)
    # backprojection
    correction = iradon(
        ratio, azi_angles, circle=False, filter_name=None
    )  # /sens_image

    axs[1, 0].imshow(mlem_rec, cmap="Greys_r")
    axs[1, 0].set_title(f"MLEM recon It={iter+1}")
    axs[1, 1].imshow(fp.T, cmap="Greys_r")
    axs[1, 1].set_title("FP of recon")
    axs[0, 2].imshow(ratio.T, cmap="Greys_r")
    axs[0, 2].set_title("Ratio Sinogram")
    axs[1, 2].imshow(correction, cmap="Greys_r")
    axs[1, 2].set_title("BP of ratio")
    plt.show()
    plt.pause(0.1)
    mlem_rec = mlem_rec * correction

plt.show(block=True)
