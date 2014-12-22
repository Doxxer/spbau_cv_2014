import cv2
import numpy as np

__author__ = 'doxer'


def get_transform_matrix(image, degrees, scale):
    rows, cols = image.shape
    image_center = (rows / 2.0, cols / 2.0)

    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, degrees, scale), [0, 0, 1]])

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    image_w2 = rows * 0.5
    image_h2 = cols * 0.5

    rotated_coordinates = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    x_coordinates = [pt[0] for pt in rotated_coordinates]
    x_pos = [x for x in x_coordinates if x > 0]
    x_neg = [x for x in x_coordinates if x < 0]

    y_coordinates = [pt[1] for pt in rotated_coordinates]
    y_pos = [y for y in y_coordinates if y > 0]
    y_neg = [y for y in y_coordinates if y < 0]

    new_w = int(abs(max(x_pos) - min(x_neg)))
    new_h = int(abs(max(y_pos) - min(y_neg)))

    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]])

    return (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :], new_w, new_h