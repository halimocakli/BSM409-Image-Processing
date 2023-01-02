import numpy as np
import cv2
import matplotlib.pyplot as plt


def local_histogram_statistics(some_image, kernel_size, k_0, k_1, k_2, k_3, C):
    """
    if k_0 * mean_global <= mean_kernel <= k_1 * mean_global and k_2 * std_global <= std_kernel <= k_3 * std_global
        g(x, y) = C * f(x, y)
    else
        g(x, y) = f(x, y)

    :param some_image: Üzerinde çalıştığımız imaj.
    :param kernel_size: İmaj üzerinde dolaştırdığımız çerçevenin boyutu.
    :param k_0: Kendimiz belirlediğimiz parametre.
    :param k_1: Kendimiz belirlediğimiz parametre.
    :param k_2: Kendimiz belirlediğimiz parametre.
    :param k_3: Kendimiz belirlediğimiz parametre.
    :param c: Kendimiz belirlediğimiz parametre.
    :return: Lokal histogram eşitliği sağlanan imaj.
    """

    some_image = some_image.astype(float)

    # Üzerinde çalışılan imajın global ortalamasını elde ediyoruz.
    mean_global = np.mean(some_image)

    # Üzerinde çalışılan imajın global standart sapmasını elde ediyoruz.
    standard_deviation_global = np.std(some_image)

    mean_lower_boundary = k_0 * mean_global
    mean_upper_boundary = k_1 * mean_global

    std_lower_boundary = k_2 * standard_deviation_global
    std_upper_boundary = k_3 * standard_deviation_global

    kernel_size = int(kernel_size / 2)
    m, n = some_image.shape[:2]

    row_indexes = []
    column_indexes = []

    for row in range(m):
        top = max(0, row - kernel_size)
        bottom = min(m, row + kernel_size + 1)

        for column in range(n):
            left = max(0, column - kernel_size)
            right = min(n, column + kernel_size + 1)

            kernel_pixels = some_image[top:bottom, left: right]
            mean_kernel = np.mean(kernel_pixels)
            standard_deviation_kernel = np.std(kernel_pixels)

            mean_condition = mean_lower_boundary <= mean_kernel <= mean_upper_boundary
            standard_deviation_condition = std_lower_boundary <= standard_deviation_kernel <= std_upper_boundary
            main_condition = mean_condition and standard_deviation_condition

            if main_condition:
                row_indexes.extend(list(range(left, right)))
                column_indexes.extend(list(range(top, bottom)))

    row_indexes = np.uint32(row_indexes)
    column_indexes = np.uint32(column_indexes)

    some_image[column_indexes, row_indexes] = C * some_image[column_indexes, row_indexes]
    return some_image.astype(np.uint8)


def main():
    # sample_array = np.uint8([[10, 20, 30, 40, 50],
    #                          [5, 15, 25, 35, 45],
    #                          [20, 30, 40, 50, 60],
    #                          [15, 25, 35, 45, 55],
    #                          [0, 10, 20, 30, 40]])

    image = cv2.imread("Images/embedded_square_noisy.tif", 0)

    kernel_size = 3
    k_0 = 0
    k_1 = 0.1
    k_2 = 0
    k_3 = 0.1
    C = 23

    # image_local_histogram_sample = local_histogram_statistics(sample_array, kernel_size, k_0, k_1, k_2, k_3, C)
    image_local_histogram = local_histogram_statistics(image, kernel_size, k_0, k_1, k_2, k_3, C)
    hrz_stacked_img = np.hstack((image, image_local_histogram))

    plt.imsave("Outputs/local_histogram_statistics_sample.jpg", hrz_stacked_img, cmap="gray")

    cv2.imshow("LOCAL HISTOGRAM STATISTICS IMAGE", hrz_stacked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
