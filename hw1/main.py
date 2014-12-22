import cv2


def show_image(window_title, image):
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_filename = "text.bmp"
source_image = cv2.imread(image_filename, 1)

gaussian_kernel_size_x = 39
gaussian_kernel_size_y = 21
blurred_image = cv2.GaussianBlur(source_image, (gaussian_kernel_size_x, gaussian_kernel_size_y), 0)
cv2.imwrite("{0}x{1}_blurred.bmp".format(str(gaussian_kernel_size_x), str(gaussian_kernel_size_y)), blurred_image)

laplacian = cv2.Laplacian(blurred_image, cv2.CV_32F, ksize=gaussian_kernel_size_y)
cv2.imwrite("{0}x{1}_laplacian.bmp".format(str(gaussian_kernel_size_x), str(gaussian_kernel_size_y)), laplacian)

result_image = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite("{0}x{1}_result_image.bmp".format(str(gaussian_kernel_size_x), str(gaussian_kernel_size_y)), result_image)

show_image("result", result_image)