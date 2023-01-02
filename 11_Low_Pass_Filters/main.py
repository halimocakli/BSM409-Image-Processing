import numpy as np
import matplotlib.pyplot as plt
import cv2


def box_filter_generator(m, n, value):
    box_filter = np.full((m, n), value)

    return box_filter / box_filter.sum()


def gaussian_filter_generator(m, n, K, sigma):
    kernel_row_range_value = m // 2
    kernel_column_range_value = n // 2

    gaussian_kernel = np.empty((m, n))

    for row in range(-kernel_row_range_value, kernel_row_range_value + 1):
        for column in range(-kernel_column_range_value, kernel_column_range_value + 1):
            r_square = row ** 2 + column ** 2
            gaussian_kernel[kernel_row_range_value + row, kernel_column_range_value + column] = K * np.exp(
                -(r_square / (2 * sigma ** 2)))

    return gaussian_kernel / gaussian_kernel.sum()


def correlation_filter_generator(input, kernel):
    operation = lambda input_kernel, kernel: (input_kernel * kernel).sum()
    return filtering(input, kernel, operation)


def filtering(input, kernel, filtering_type):
    pass


if __name__ == '__main__':
    gaussian_filter_generator(3, 3, 1, 5)
