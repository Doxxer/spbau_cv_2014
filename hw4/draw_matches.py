import cv2
import numpy as np


def draw_matches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    result = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    result[:rows1, :cols1, :] = np.dstack([img1, img1, img1])
    result[:rows2, cols1:cols1 + cols2, :] = np.dstack([img2, img2, img2])

    for match in matches:
        (x1, y1) = kp1[match.queryIdx].pt
        (x2, y2) = kp2[match.trainIdx].pt

        cv2.circle(result, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(result, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)
        cv2.line(result, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 0, 255), 1)
    return result