import numpy as np
import cv2

INPUT_IMAGE_FILENAME = "text.bmp"


def show_image(window_title, image):
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def make_kernel(w, h):
    return np.ones((w, h), np.uint8)


def save(dilated, eroded, number):
    cv2.imwrite("{0}_dilated.bmp".format(str(number)), dilated)
    cv2.imwrite("{0}_eroded.bmp".format(str(number)), eroded)


def find_binary_connected_components(image):
    height, width = image.shape

    image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("thresholded.bmp", image)

    # OpenCV Error: Sizes of input arguments do not match
    # (mask must be 2 pixel wider and 2 pixel taller than filled image) in cvFloodFill
    mask = np.zeros((height + 2, width + 2), np.uint8)
    resulting_rectangles = []
    for j in range(0, height):
        for i in range(0, width):
            if image[j, i] != 0:
                continue
            result_flag, rect = cv2.floodFill(image, mask, (i, j), 0, flags=cv2.FLOODFILL_MASK_ONLY)
            if result_flag:
                resulting_rectangles.append(rect)
    return resulting_rectangles


def draw_rectangles(image, rectangles):
    for rectangle in rectangles:
        x, y, w, h = rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)


if __name__ == "__main__":
    source_image = cv2.imread(INPUT_IMAGE_FILENAME, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    dilated_image, eroded_image = source_image, source_image

    dilated_image = cv2.dilate(dilated_image, make_kernel(1, 5))
    eroded_image = cv2.erode(eroded_image, make_kernel(1, 3))
    save(dilated_image, eroded_image, 1)

    dilated_image = cv2.dilate(dilated_image, make_kernel(3, 1))
    eroded_image = cv2.erode(eroded_image, make_kernel(4, 1))
    save(dilated_image, eroded_image, 2)

    rectangles = find_binary_connected_components(eroded_image)

    color_image = cv2.imread(INPUT_IMAGE_FILENAME, cv2.CV_LOAD_IMAGE_COLOR)
    draw_rectangles(color_image, rectangles)
    cv2.imwrite("result.bmp", color_image)