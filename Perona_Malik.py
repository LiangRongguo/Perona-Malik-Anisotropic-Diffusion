import numpy as np
from PIL import Image
import math

def f(lamb, b):
    """
    :param lamb:
    :param b:
    :return:
    """
    return 1 / (1 + ((lamb / b) ** 2))


def anisotropic_diffusion(image_location, log_freq, iterations, b, lamb=0.01):
    """
    :param image_location:
    :param log_freq:
    :param iterations:
    :param b:
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
            f(I_North, b) * I_North +
            f(I_South, b) * I_South +
            f(I_East, b) * I_East +
            f(I_West, b) * I_West
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


def test():
    log = anisotropic_diffusion("images/gaussian_noise_dog.jpg", log_freq=1, iterations=10, b=0.1, lamb=0.1)


if __name__ == "__main__":
    test()
