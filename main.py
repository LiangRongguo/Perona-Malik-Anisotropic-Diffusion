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

    figure.tight_layout()
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

    figure.tight_layout()
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

    image_x = ndimage.filters.convolve(image / 255, edge_kernel_x)
    image_y = ndimage.filters.convolve(image / 255, edge_kernel_y)
    image_edge = np.hypot(image_x, image_y)

    return image_edge


def PDE():
    # parameters
    num_col = 5
    iterations = 80
    k_gaussian = 0.1
    lamb_gaussian = 0.1
    k_sp = 0.1
    lamb_sp = 0.1
    k_sp1 = 0.2

    print("applying PDE with Gaussian noise image...")

    log_Gaussian_PDE = anisotropic_diffusion(GAUSSIAN_IMAGE, log_freq=iterations / (num_col - 1), iterations=iterations,
                                             k=k_gaussian, lamb=lamb_gaussian)

    figure = plt.figure()
    plt.title("PDE on Gaussian Noise Image with k = " + str(k_gaussian))
    plt.axis('off')
    plt.gray()

    for i in range(len(log_Gaussian_PDE)):
        ax1 = figure.add_subplot(2, num_col, i + 1)
        plt.title("t=" + str(iterations / (num_col - 1) * i))
        plt.axis('off')
        ax1.imshow(log_Gaussian_PDE[i])

        ax2 = figure.add_subplot(2, num_col, i + num_col + 1)
        plt.title("edge")
        plt.axis('off')
        ax2.imshow(edge_detection(log_Gaussian_PDE[i]))

    plt.savefig("images/PDE_Gaussian.jpg")
    plt.show()
    print("\tPDE with Gaussian noise image saved in: images/PDE_Gaussian.jpg\n")

    #######################################################

    print("applying PDE with Salt & Pepper noise image (k = 0.1)...")

    iterations_sp = 160

    log_SA_PDE = anisotropic_diffusion(SALT_PEPPER_IMAGE, log_freq=iterations_sp / (num_col - 1),
                                       iterations=iterations_sp,
                                       k=k_sp, lamb=lamb_sp)

    figure = plt.figure()
    plt.title("PDE on Salt & Pepper Noise Image with k = " + str(k_sp))
    plt.axis('off')
    plt.gray()

    for i in range(len(log_SA_PDE)):
        ax1 = figure.add_subplot(2, num_col, i + 1)
        plt.title("t=" + str(iterations_sp / (num_col - 1) * i))
        plt.axis('off')
        ax1.imshow(log_SA_PDE[i])

        ax2 = figure.add_subplot(2, num_col, i + num_col + 1)
        plt.axis('off')
        plt.title("edge")
        ax2.imshow(edge_detection(log_SA_PDE[i]))

    plt.savefig("images/PDE_SP.jpg")
    plt.show()
    print("\tPDE with Salt & Pepper noise (k=0.1) image saved in: images/PDE_SP.jpg\n")

    #######################################################

    print("applying PDE with Salt & Pepper noise image (k = 0.2)...")

    log_SA_PDE1 = anisotropic_diffusion(SALT_PEPPER_IMAGE, log_freq=iterations / (num_col - 1), iterations=iterations,
                                        k=k_sp1, lamb=lamb_sp)

    figure = plt.figure()
    plt.title("PDE on Salt & Pepper Noise Image with k = " + str(k_sp1))
    plt.axis('off')
    plt.gray()

    for i in range(len(log_SA_PDE1)):
        ax1 = figure.add_subplot(2, num_col, i + 1)
        plt.title("t=" + str(iterations / (num_col - 1) * i))
        plt.axis('off')
        ax1.imshow(log_SA_PDE1[i])

        ax2 = figure.add_subplot(2, num_col, i + num_col + 1)
        plt.axis('off')
        plt.title("edge")
        ax2.imshow(edge_detection(log_SA_PDE1[i]))

    plt.savefig("images/PDE_SP1.jpg")
    plt.show()
    print("\tPDE with Salt & Pepper noise (k=0.2) image saved in: images/PDE_SP1.jpg\n")

    #######################################################

    print("analyzing the PSNR of Gaussian/S&P Noise...")

    original_image = np.array(Image.open(ORIGINAL_IMAGE).convert('L')) / 255

    PSNR_Gaussian = []
    for target in log_Gaussian_PDE:
        PSNR_Gaussian.append(PSNR(original_image, target))
    print("\tdone with calculating the PSNR result for Gaussian noise")

    PSNR_SP = []
    for target in log_SA_PDE:
        PSNR_SP.append(PSNR(original_image, target))
    print("\tdone with calculating the PSNR result for Salt & Pepper noise with k = 0.1")

    PSNR_SP1 = []
    for target in log_SA_PDE1:
        PSNR_SP1.append(PSNR(original_image, target))
    print("\tdone with calculating the PSNR result for Salt & Pepper noise with k = 0.2")

    figure = plt.figure()
    x = np.arange(0, iterations + iterations / (num_col - 1), iterations / (num_col - 1))
    x1 = np.arange(0, iterations_sp + iterations_sp / (num_col - 1), iterations_sp / (num_col - 1))

    ax1 = figure.add_subplot(3, 1, 1)
    ax1.plot(x, PSNR_Gaussian, ".-")
    plt.xlabel("Iterations")
    plt.ylabel("PSNR/dB")
    plt.title("PSNR for Gaussian Noise image with k = 0.1")

    ax2 = figure.add_subplot(3, 1, 2)
    ax2.plot(x1, PSNR_SP, ".-")
    plt.xlabel("Iterations")
    plt.ylabel("PSNR/dB")
    plt.title("PSNR for Salt & Pepper Noise image with k = 0.1")

    ax3 = figure.add_subplot(3, 1, 3)
    ax3.plot(x1, PSNR_SP, ".-")
    plt.xlabel("Iterations")
    plt.ylabel("PSNR/dB")
    plt.title("PSNR for Salt & Pepper Noise image with k = 0.2")

    figure.tight_layout()
    plt.savefig('images/PSNR_result.jpg')
    print("\tPSNR result save in: images/PSNR_result.jpg")
    print("\tPSNR for Gaussian with k = 0.1 in iterations", x, "is: ")
    print("\t\t", PSNR_Gaussian)
    print("\tPSNR for Salt & Pepper with k = 0.1 in iterations", x, "is: ")
    print("\t\t", PSNR_SP)
    print("\tPSNR for Salt & Pepper with k = 0.2 in iterations", x, "is: ")
    print("\t\t", PSNR_SP1)

    plt.savefig("images/PDE_PSNR.jpg")
    plt.show()

    print("")


def compare_k():
    print("comparing the effects of different k...")

    iterations_gaussian = 40
    iterations_sp = 100
    k = np.arange(0.05, 0.61, 0.01)
    lamb = 0.1

    result_image_gaussian = []
    result_psnr_gaussian = []

    result_image_sp = []
    result_psnr_sp = []

    original_image = np.array(Image.open(ORIGINAL_IMAGE).convert('L')) / 255

    for each_k in k:
        log = anisotropic_diffusion(GAUSSIAN_IMAGE, log_freq=iterations_gaussian,
                                    iterations=iterations_gaussian,
                                    k=each_k, lamb=lamb)
        result_image_gaussian.append(log[-1])
        result_psnr_gaussian.append(PSNR(log[-1], original_image))

        log = anisotropic_diffusion(SALT_PEPPER_IMAGE, log_freq=iterations_sp,
                                    iterations=iterations_sp,
                                    k=each_k, lamb=lamb)
        result_image_sp.append(log[-1])
        result_psnr_sp.append(PSNR(log[-1], original_image))

    print("\tPSNR for gaussian:", result_psnr_gaussian)
    print("\tPSNR for salt & pepper", result_psnr_sp)

    figure = plt.figure()
    plt.axis('off')
    plt.gray()

    ax1 = figure.add_subplot(2, 1, 1)
    ax1.plot(k, result_psnr_gaussian, ".-")
    plt.xlabel("k")
    plt.ylabel("PSNR/dB")
    plt.title("PSNR for Gaussian Noise image with 40 iterations")

    ax2 = figure.add_subplot(2, 1, 2)
    ax2.plot(k, result_psnr_sp, ".-")
    plt.xlabel("k")
    plt.ylabel("PSNR/dB")
    plt.title("PSNR for Salt & Pepper Noise image with 100 iterations")

    figure.tight_layout()
    plt.savefig("images/PDE_k.jpg")
    plt.show()

    print("")


def split_PDE():
    print("splitting the image to apply PDE respectively...")

    original_image = np.array(Image.open(ORIGINAL_IMAGE).convert('L')) / 255

    gaussian_image = np.array(Image.open(GAUSSIAN_IMAGE).convert('L')) / 255
    gaussian_pde_result = anisotropic_diffusion(GAUSSIAN_IMAGE, log_freq=40, iterations=40, k=0.1, lamb=0.1)[-1]

    salt_pepper_image = np.array(Image.open(SALT_PEPPER_IMAGE).convert('L')) / 255
    salt_pepper_pde_result = anisotropic_diffusion(GAUSSIAN_IMAGE, log_freq=100, iterations=100, k=0.1, lamb=0.1)[-1]

    gaussian_pde_result_2x2 = np.zeros(original_image.shape, np.float64)
    gaussian_pde_result_4x4 = np.zeros(original_image.shape, np.float64)
    salt_pepper_pde_result_2x2 = np.zeros(original_image.shape, np.float64)
    salt_pepper_pde_result_4x4 = np.zeros(original_image.shape, np.float64)

    h = original_image.shape[0]
    w = original_image.shape[1]

    div = 2
    unit_h = int(h / div)
    unit_w = int(w / div)

    for i in range(div):
        for j in range(div):
            gaussian_part = gaussian_image[i * unit_h: (i + 1) * unit_h, j * unit_w: (j + 1) * unit_w]
            gaussian_filename = "./images/split" + str(div) + "x" + str(div) + "/gaussian" + str(i) + str(j) + ".jpg"
            imageio.imwrite(gaussian_filename, np.uint8(gaussian_part * 255))
            gaussian_pde_result_2x2[i * unit_h: (i + 1) * unit_h, j * unit_w: (j + 1) * unit_w] = \
                anisotropic_diffusion(gaussian_filename, log_freq=40, iterations=40, k=0.1, lamb=0.1)[-1]

            salt_pepper_part = salt_pepper_image[i * unit_h: (i + 1) * unit_h, j * unit_w: (j + 1) * unit_w]
            salt_pepper_filename = "./images/split" + str(div) + "x" + str(div) + "/salt_pepper" + str(i) + str(
                j) + ".jpg"
            imageio.imwrite(salt_pepper_filename, np.uint8(salt_pepper_part * 255))
            salt_pepper_pde_result_2x2[i * unit_h: (i + 1) * unit_h, j * unit_w: (j + 1) * unit_w] = \
                anisotropic_diffusion(salt_pepper_filename, log_freq=100, iterations=100, k=0.1, lamb=0.1)[-1]

    div = 4
    unit_h = int(h / div)
    unit_w = int(w / div)

    for i in range(div):
        for j in range(div):
            gaussian_part = gaussian_image[i * unit_h: (i + 1) * unit_h, j * unit_w: (j + 1) * unit_w]
            gaussian_filename = "./images/split" + str(div) + "x" + str(div) + "/gaussian" + str(i) + str(j) + ".jpg"
            imageio.imwrite(gaussian_filename, np.uint8(gaussian_part * 255))
            gaussian_pde_result_4x4[i * unit_h: (i + 1) * unit_h, j * unit_w: (j + 1) * unit_w] = \
                anisotropic_diffusion(gaussian_filename, log_freq=40, iterations=40, k=0.1, lamb=0.1)[-1]

            salt_pepper_part = salt_pepper_image[i * unit_h: (i + 1) * unit_h, j * unit_w: (j + 1) * unit_w]
            salt_pepper_filename = "./images/split" + str(div) + "x" + str(div) + "/salt_pepper" + str(i) + str(
                j) + ".jpg"
            imageio.imwrite(salt_pepper_filename, np.uint8(salt_pepper_part * 255))
            salt_pepper_pde_result_4x4[i * unit_h: (i + 1) * unit_h, j * unit_w: (j + 1) * unit_w] = \
                anisotropic_diffusion(salt_pepper_filename, log_freq=100, iterations=100, k=0.1, lamb=0.1)[-1]

    psnr_gaussian = round(PSNR(gaussian_image, original_image), 4)
    psnr_gaussian_1x1 = round(PSNR(gaussian_pde_result, original_image), 4)
    psnr_gaussian_2x2 = round(PSNR_split(gaussian_pde_result_2x2, original_image, gaussian_pde_result), 4)
    psnr_gaussian_4x4 = round(PSNR_split(gaussian_pde_result_4x4, original_image, gaussian_pde_result), 4)

    psnr_salt_pepper = round(PSNR(salt_pepper_image, original_image), 4)
    psnr_salt_pepper_1x1 = round(PSNR(salt_pepper_pde_result, original_image), 4)
    psnr_salt_pepper_2x2 = round(PSNR_split(salt_pepper_pde_result_2x2, original_image, salt_pepper_pde_result), 4)
    psnr_salt_pepper_4x4 = round(PSNR_split(salt_pepper_pde_result_4x4, original_image, salt_pepper_pde_result), 4)

    print("\tPSNR for gaussian noise image:", str(psnr_gaussian))
    print("\tPSNR for PDE on gaussian noise image with split 1x1:", str(psnr_gaussian_1x1))
    print("\tPSNR for PDE on gaussian noise image with split 2x2:", str(psnr_gaussian_2x2))
    print("\tPSNR for PDE on gaussian noise image with split 4x4:", str(psnr_gaussian_4x4))

    print("")

    print("\tPSNR for salt & pepper noise image:", str(psnr_salt_pepper))
    print("\tPSNR for PDE on salt & pepper noise image with split 1x1:", str(psnr_salt_pepper_1x1))
    print("\tPSNR for PDE on salt & pepper noise image with split 2x2:", str(psnr_salt_pepper_2x2))
    print("\tPSNR for PDE on salt & pepper noise image with split 4x4:", str(psnr_salt_pepper_4x4))

    figure = plt.figure()
    plt.title("PSNR for different split")
    plt.axis('off')
    plt.gray()

    ax1 = figure.add_subplot(2, 4, 1)
    plt.title("Gaussian")
    plt.text(166., 320., "PSNR: " + str(round(psnr_gaussian, 2)), size=10, ha="center", va="top", )
    plt.axis('off')
    ax1.imshow(gaussian_image)

    ax2 = figure.add_subplot(2, 4, 2)
    plt.title("1 x 1")
    plt.text(166., 320., "PSNR: " + str(round(psnr_gaussian_1x1, 2)), size=10, ha="center", va="top",)
    plt.axis('off')
    ax2.imshow(gaussian_pde_result)

    ax3 = figure.add_subplot(2, 4, 3)
    plt.title("2 x 2")
    plt.text(166., 320., "PSNR: " + str(round(psnr_gaussian_2x2, 2)), size=10, ha="center", va="top", )
    plt.axis('off')
    ax3.imshow(gaussian_pde_result_2x2)

    ax4 = figure.add_subplot(2, 4, 4)
    plt.title(psnr_gaussian_4x4)
    plt.title("4 x 4")
    plt.text(166., 320., "PSNR: " + str(round(psnr_gaussian_4x4, 2)), size=10, ha="center", va="top", )
    plt.axis('off')
    ax4.imshow(gaussian_pde_result_4x4)

    ax5 = figure.add_subplot(2, 4, 5)
    plt.title("Salt & Pepper")
    plt.text(166., 320., "PSNR: " + str(round(psnr_salt_pepper, 2)), size=10, ha="center", va="top", )
    plt.axis('off')
    ax5.imshow(salt_pepper_image)

    ax6 = figure.add_subplot(2, 4, 6)
    plt.title("1 x 1")
    plt.text(166., 320., "PSNR: " + str(round(psnr_salt_pepper_1x1, 2)), size=10, ha="center", va="top", )
    plt.axis('off')
    ax6.imshow(salt_pepper_pde_result)

    ax7 = figure.add_subplot(2, 4, 7)
    plt.title("2 x 2")
    plt.text(166., 320., "PSNR: " + str(round(psnr_salt_pepper_2x2, 2)), size=10, ha="center", va="top", )
    plt.axis('off')
    ax7.imshow(salt_pepper_pde_result_2x2)

    ax8 = figure.add_subplot(2, 4, 8)
    plt.title("4 x 4")
    plt.text(166., 320., "PSNR: " + str(round(psnr_salt_pepper_4x4, 2)), size=10, ha="center", va="top", )
    plt.axis('off')
    ax8.imshow(salt_pepper_pde_result_4x4)

    plt.savefig("./images/split_PDE.jpg")
    plt.show()


def main():
    print("\nThis is the ECE6560 course project...")
    print("\tName: \tRongguo Liang\n")

    add_noise()
    gaussian_low_pass_filter_and_edge_detection()
    PDE()
    compare_k()
    split_PDE()

    print("\ndone...")


if __name__ == "__main__":
    main()
