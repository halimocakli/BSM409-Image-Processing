import cv2
import numpy as np
import matplotlib.pyplot as plt

L = 256  # 8 BITS CONSTANT VALUE


def histogram_equalization(_image):
    histogram_equalized_image = cv2.equalizeHist(_image)

    histogram, bins = np.histogram(histogram_equalized_image, bins=256, range=(0, 256))

    adjoined = np.hstack((_image, histogram_equalized_image))

    cv2.imshow("HIST EQUALIZED IMAGE", adjoined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread("./Images/top_left.tif", 0)
    histogram_equalization(image)
