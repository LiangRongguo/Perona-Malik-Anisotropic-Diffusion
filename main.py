import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.util import random_noise
from PIL import Image
import imageio
from Perona_Malik import *

# Original Image
ORIGINAL_IMAGE = "images/dog.jpg"
# Gaussian Noise Image
GAUSSIAN_IMAGE = "images/gaussian_noise_dog.jpg"
# Salt & Pepper Image
SALT_PEPPER_IMAGE = "images/salt_pepper_dog.jpg"
# Gaussian Noise Filtered Image
GAUSSIAN_FILTERED_IMAGE = "images/gaussian_noise_filtered_dog.jpg"
# Salt & Pepper Noise Filtered Image
SALT_PEPPER__FILTERED_IMAGE = "images/salt_pepper_filtered_dog.jpg"
# Gaussian Noise Filtered Image - Edge
GAUSSIAN_FILTERED_IMAGE_EDGE = "images/gaussian_noise_filtered_dog_edge.jpg"
# Salt & Pepper Noise Filtered Image - Edge
SALT_PEPPER__FILTERED_IMAGE_EDGE = "images/salt_pepper_filtered_dog_edge.jpg"


def add_noise():
    """
        load original image
        then add noise with gaussian and salt & pepper
    """

    print("adding noise...")

    figure = plt.figure()
    plt.gray()

    image = np.array(Image.open(ORIGINAL_IMAGE).convert('L'))
    gaussian_image = random_noise(image=image, mode='gaussian', seed=None, clip=True)
    salt_pepper_image = random_noise(image=image, mode='s&p', seed=None, clip=True)

    ax1 = figure.add_subplot(131)
    plt.title("Original Image")
    plt.axis('off')
    ax2 = figure.add_subplot(132)
    plt.title("Gaussian Noise Image")
    plt.axis('off')
    ax3 = figure.add_subplot(133)
    plt.title("Salt & Pepper Noise Image")
    plt.axis('off')

    ax1.imshow(image)
    ax2.imshow(gaussian_image)
    ax3.imshow(salt_pepper_image)

    plt.show()

    imageio.imwrite(GAUSSIAN_IMAGE, np.uint8(gaussian_image * 255))
    print("\tImage with Gaussian noise saved in: " + GAUSSIAN_IMAGE)
    imageio.imwrite(SALT_PEPPER_IMAGE, np.uint8(salt_pepper_image * 255))
    print("\tImage with Salt & Pepper noise saved in: " + SALT_PEPPER_IMAGE)

    # done with adding noise to images


def gaussian_low_pass_filter_and_edge_detection():
    """
        load two kinds of noise images
        apply simple gaussian filter
        and detect edges
    """
    print("\napplying Gaussian low pass filter...")

    figure = plt.figure()

    gaussian_image = np.array(Image.open(GAUSSIAN_IMAGE).convert('L'))
    salt_pepper_image = np.array(Image.open(SALT_PEPPER_IMAGE).convert('L'))

    gaussian_image_filtered = ndimage.gaussian_filter(gaussian_image, sigma=5)
    salt_pepper_image_filtered = ndimage.gaussian_filter(salt_pepper_image, sigma=5)

    imageio.imwrite(GAUSSIAN_FILTERED_IMAGE, gaussian_image_filtered)
    print("\tImage with Gaussian noise filtered saved in: " + GAUSSIAN_FILTERED_IMAGE)
    imageio.imwrite(SALT_PEPPER__FILTERED_IMAGE, salt_pepper_image_filtered)
    print("\tImage with S&A noise filtered saved in: " + SALT_PEPPER__FILTERED_IMAGE)

    print("\napplying Edge Detection...")

    gaussian_image_filtered_edge = edge_detection(gaussian_image_filtered)
    salt_pepper_image_filtered_edge = edge_detection(salt_pepper_image_filtered)

    imageio.imwrite(GAUSSIAN_FILTERED_IMAGE_EDGE, np.uint8(gaussian_image_filtered_edge * 255))
    print("\tImage with edges of Gaussian noise filtered saved in: " + GAUSSIAN_FILTERED_IMAGE_EDGE)
    imageio.imwrite(SALT_PEPPER__FILTERED_IMAGE_EDGE, np.uint8(salt_pepper_image_filtered_edge * 255))
    print("\tImage with edges of S&A noise filtered saved in: " + SALT_PEPPER__FILTERED_IMAGE_EDGE)

    ax1 = figure.add_subplot(321)
    plt.title("Gaussian Noise Image")
    plt.axis('off')
    ax2 = figure.add_subplot(322)
    plt.title("Salt & Pepper Noise Image")
    plt.axis('off')
    ax3 = figure.add_subplot(323)
    plt.title("Gaussian Noise Image Filtered")
    plt.axis('off')
    ax4 = figure.add_subplot(324)
    plt.title("Salt & Pepper Noise Image Filtered")
    plt.axis('off')
    ax5 = figure.add_subplot(325)
    plt.title("Edges of Gaussian Filtered image")
    plt.axis('off')
    ax6 = figure.add_subplot(326)
    plt.title("Edges of S&A Filtered image")
    plt.axis('off')

    ax1.imshow(gaussian_image)
    ax2.imshow(salt_pepper_image)
    ax3.imshow(gaussian_image_filtered)
    ax4.imshow(salt_pepper_image_filtered)
    ax5.imshow(gaussian_image_filtered_edge)
    ax6.imshow(salt_pepper_image_filtered_edge)

    plt.show()

    print("")


def edge_detection(image):
    """
    :param image: original image
    :return: detected edges of input image
    """

    edge_kernel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], np.int32)

    edge_kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], np.int32)

    image_x = ndimage.filters.convolve(image/255, edge_kernel_x)
    image_y = ndimage.filters.convolve(image/255, edge_kernel_y)
    image_edge = np.hypot(image_x, image_y)

    return image_edge


def PDE():
    print("applying PDE with Gaussian noise image...")

    log_Gaussian_PDE = anisotropic_diffusion(GAUSSIAN_IMAGE, log_freq=1, iterations=4, b=0.1, lamb=0.1)

    figure = plt.figure()
    plt.gray()

    for i in range(len(log_Gaussian_PDE)):
        ax1 = figure.add_subplot(2, 5, i + 1)
        plt.title("t="+str(20 * i))
        plt.axis('off')
        ax1.imshow(log_Gaussian_PDE[i])

        ax2 = figure.add_subplot(2, 5, i + 6)
        plt.title("edge")
        plt.axis('off')
        ax2.imshow(edge_detection(log_Gaussian_PDE[i]))

    plt.savefig("images/PDE_Gaussian.jpg")
    plt.show()
    print("\tPDE with Gaussian noise image saved in: images/PDE_Gaussian.jpg\n")

    #######################################################

    print("applying PDE with Salt & Pepper noise image...")

    log_SA_PDE = anisotropic_diffusion(SALT_PEPPER_IMAGE, log_freq=20, iterations=80, b=0.1, lamb=0.1)

    figure = plt.figure()
    plt.gray()

    for i in range(len(log_SA_PDE)):
        ax1 = figure.add_subplot(2, 5, i + 1)
        plt.title("t="+str(20 * i))
        plt.axis('off')
        ax1.imshow(log_SA_PDE[i])

        ax2 = figure.add_subplot(2, 5, i + 6)
        plt.axis('off')
        plt.title("edge")
        ax2.imshow(edge_detection(log_SA_PDE[i]))

    plt.savefig("images/PDE_SP.jpg")
    plt.show()
    print("\tPDE with Salt & Pepper noise image saved in: images/PDE_SP.jpg\n")

    image = np.array(Image.open(ORIGINAL_IMAGE).convert('L')) / 255
    for target in log_Gaussian_PDE:
        print(PSNR(target, image))


def main():
    # add_noise()
    # gaussian_low_pass_filter_and_edge_detection()
    PDE()


if __name__ == "__main__":
    print("\nThis is the ECE6560 course project...")
    print("\tName: \tRongguo Liang")
    print("\tGT id: \trliang37\n")

    main()
