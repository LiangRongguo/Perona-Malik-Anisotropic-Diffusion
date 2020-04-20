import numpy as np
from PIL import Image
import math


def coefficient(I, k):
    """
    :param I:
    :param k:
    :return:
    """
    return 1 / (1 + ((I / k) ** 2))


def anisotropic_diffusion(image_location, log_freq, iterations, k, lamb=0.01):
    """
    :param image_location:
    :param log_freq:
    :param iterations:
    :param k:
    :param lamb:
    :return:
    """
    image = np.array(Image.open(image_location).convert('L')) / 255
    new_image = np.zeros(image.shape, dtype=image.dtype)

    result = [image]

    for t in range(iterations):
        I_North = image[:-2, 1:-1] - image[1:-1, 1:-1]
        I_South = image[2:, 1:-1] - image[1:-1, 1:-1]
        I_East = image[1:-1, 2:] - image[1:-1, 1:-1]
        I_West = image[1:-1, :-2] - image[1:-1, 1:-1]

        new_image[1:-1, 1:-1] = image[1:-1, 1:-1] + lamb * (
            coefficient(I_North, k) * I_North +
            coefficient(I_South, k) * I_South +
            coefficient(I_East, k) * I_East +
            coefficient(I_West, k) * I_West
        )

        image = new_image

        if (t+1) % log_freq == 0:
            result.append(image.copy())

    return result


def PSNR(target, ref):
    """
    compute the PSNR between result image and original image
    :param target: type float64
    :param ref: type float64
    :return:
    """
    mse = np.mean((target - ref) ** 2)

    if mse == 0:
        return 100

    max_val = 1.0

    return 20 * math.log10(max_val / math.sqrt(mse))


def PSNR_split(target, ref, original_pde_result):
    """
    compute the PSNR between result image and original image
    :param target: type float64
    :param ref: type float64
    :param original_pde_result:
    :return:
    """
    h = target.shape[0]
    w = target.shape[1]

    target_copy = target.copy()

    for i in range(h):
        for j in range(w):
            if target_copy[i, j] == 0:
                target_copy[i, j] = original_pde_result[i, j]

    mse = np.mean((target_copy - ref) ** 2)

    if mse == 0:
        return 100

    max_val = 1.0

    return 20 * math.log10(max_val / math.sqrt(mse))


def test():
    anisotropic_diffusion("images/gaussian_noise_dog.jpg", log_freq=1, iterations=10, k=0.1, lamb=0.1)


if __name__ == "__main__":
    test()
