import cv2
import numpy as np
import matplotlib.pyplot as plt


def horizontal_stacking(*args):
    return np.hstack(args)


def vertical_stacking(*args):
    return np.vstack(args)


def rescale_to_8bits(image):
    s = image.astype(float)
    s = (((s - np.min(s)) / (np.max(s))) * 255).astype(np.uint8)


def histogram_equalization_with_built_in_function():
    image = cv2.imread("./Images/top_left.tif", 0)
    histogram_equalized_image = cv2.equalizeHist(image)

    adjoined = np.hstack((image, histogram_equalized_image))

    cv2.imshow("HIST EQUALIZED IMAGE", adjoined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    histogram_equalization_with_built_in_function()
