import cv2
import numpy as np
import matplotlib.pyplot as plt


def horizontal_stacking(*args):
    return np.hstack(args)


def binary_intensity_slicing(image, gap_a, gap_b, low_intensity, high_intensity):
    output_image = np.full_like(image, low_intensity)
    index_boolean = np.logical_and(image > gap_a, image < gap_b)
    output_image[index_boolean] = high_intensity
    return output_image


def linear_intensity_slicing(image, gap_a, gap_b, intensity):
    output_image = image.copy()
    index_boolean = np.logical_and(image > gap_a, image < gap_b)
    output_image[index_boolean] = intensity
    return output_image


if __name__ == '__main__':
    kidney_image = cv2.imread("./Images/kidney.tif", 0)

    gap_a = 150
    gap_b = 200
    low_intensity = 10
    high_intensity = 255
    linear_image_intensity = 255

    binary_image = binary_intensity_slicing(image=kidney_image, gap_a=gap_a, gap_b=gap_b, low_intensity=low_intensity,
                                            high_intensity=high_intensity)

    linear_image = linear_intensity_slicing(image=kidney_image, gap_a=gap_a, gap_b=gap_b,
                                            intensity=linear_image_intensity)

    binary_linear_adjoined = horizontal_stacking(kidney_image, binary_image, linear_image)

    plt.imshow(binary_linear_adjoined, cmap="gray")
    plt.show()

    cv2.imshow("ORIGINAL-BINARY-LINEAR INTENSITY SLICED IMAGES", binary_linear_adjoined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.imsave("./Outputs/binary_linear_intensity_sliced_kidney.jpg", binary_linear_adjoined, cmap="gray")
