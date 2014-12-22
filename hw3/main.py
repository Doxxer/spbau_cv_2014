import cv2
import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift

INPUT_IMAGE_FILENAME = "image.bmp"


def save_image(image, info):
    cv2.imwrite("image_{0}.bmp".format(info), image)


def fourier_hpf(image, zeroing_size):
    fshift = fftshift(fft2(image))

    # HPF
    rows, cols = image.shape
    center_row, center_col = rows / 2, cols / 2
    fshift[center_row - zeroing_size:center_row + zeroing_size, center_col - zeroing_size:center_col + zeroing_size] = 0

    # inverse
    img_back = ifft2(ifftshift(fshift))
    img_back = numpy.abs(img_back)
    save_image(img_back, "fourier_{0}".format(zeroing_size))


def laplacian_hpf(image, kernel_size):
    laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=kernel_size)
    save_image(laplacian, "laplacian_{0}".format(kernel_size))


if __name__ == "__main__":
    source_image = cv2.imread(INPUT_IMAGE_FILENAME, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    fourier_hpf(source_image, 30)
    laplacian_hpf(source_image, 3)