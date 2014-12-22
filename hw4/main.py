import cv2
import numpy as np

from draw_matches import draw_matches
from rotate_image_without_cropping import get_transform_matrix


INPUT_IMAGE_FILENAME = "image.bmp"


def save_image(image, info):
    cv2.imwrite("image_{0}.bmp".format(info), image)


def find_matches(descr1, descr2):
    flann = (cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})).knnMatch(descr1,
                                                                                           descr2, k=2)
    matches_mask = [0] * len(flann)
    for index, (a, b) in enumerate(flann):
        if a.distance < 0.75 * b.distance:
            matches_mask[index] = 1
    return [match[0] for (index, match) in enumerate(flann) if matches_mask[index] == 1]


def calc_precision(kp1, kp2, _matches, transform_mat):
    result = 0
    for match in _matches:
        x = np.append(kp1[match.queryIdx].pt, 1)
        transformed_point = np.array([np.dot(x, np.array(transform_mat[0, :])[0]),
                                      np.dot(x, np.array(transform_mat[1, :])[0])])
        distance = np.linalg.norm(transformed_point - kp2[match.trainIdx].pt)
        result += 1 if distance < 3 else 0
    return result


if __name__ == "__main__":
    # load images
    source_image = cv2.imread(INPUT_IMAGE_FILENAME, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    transform_matrix, new_w, new_h = get_transform_matrix(source_image, 45, 0.5)
    transformed_image = cv2.warpAffine(source_image, transform_matrix, (new_w, new_h))

    # detect POIs
    sift = cv2.SIFT(500)
    kp_source, descr_source = sift.detectAndCompute(source_image, None)
    kp_transformed, descr_transformed = sift.detectAndCompute(transformed_image, None)

    # draw POIs
    source_kp_image = cv2.drawKeypoints(source_image, kp_source,
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    transformed_kp_image = cv2.drawKeypoints(transformed_image, kp_transformed,
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # find matches
    matches = find_matches(descr_source, descr_transformed)

    # calc precision
    match_count = calc_precision(kp_source, kp_transformed, matches, transform_matrix)
    print "Match precision = %s%% (%s from %s)" % (match_count * 100.0 / len(matches), match_count, len(matches))

    # draw matches
    result_image = draw_matches(source_image, kp_source, transformed_image, kp_transformed, matches)

    # save images
    save_image(source_kp_image, 'source_kp')
    save_image(transformed_kp_image, 'transformed_kp')
    save_image(result_image, 'result')
