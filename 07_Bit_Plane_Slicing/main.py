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
    return s


# 0 <= bit_plane <= 7
def bit_plane_slicing(image, bit_plane):
    bit_plane_image = np.full_like(image, 2 ** bit_plane)
    bit_plane_sliced_image = np.bitwise_and(image, bit_plane_image)
    return bit_plane_sliced_image


def image_compressor(image, bit_planes):
    compressed_image = np.zeros_like(image)

    for bit_plane in bit_planes:
        compressed_image = compressed_image + bit_plane_slicing(image, bit_plane)

    return compressed_image


if __name__ == "__main__":
    image_100_dollars = cv2.imread("./Images/100-dollars.tif", 0)
    image_fractal_iris = cv2.imread("./Images/fractal-iris.tif", 0)
    image_headct_vandy = cv2.imread("./Images/headCT-Vandy.tif", 0)

    images = [image_100_dollars, image_fractal_iris, image_headct_vandy]
    image_names = ["image_100_dollars", "image_fractal_iris", "image_headct_vandy"]
    bit_plane_sliced_images = []

    cnt = 0
    for image in images:
        for bit_plane in range(7, -1, -1):
            bit_plane_sliced_image = bit_plane_slicing(image, bit_plane)

            print(
                f"Bit Plane is {bit_plane} and Unique values of {image_names[cnt]} is {np.unique(bit_plane_sliced_image)}")

            bit_plane_sliced_image = rescale_to_8bits(bit_plane_sliced_image)
            bit_plane_sliced_images.append(bit_plane_sliced_image)

        print("")

        bit_plane_sliced_images = bit_plane_sliced_images[::-1]

        row_1 = horizontal_stacking(image, bit_plane_sliced_images[0], bit_plane_sliced_images[1])
        row_2 = horizontal_stacking(bit_plane_sliced_images[2], bit_plane_sliced_images[3], bit_plane_sliced_images[4])
        row_3 = horizontal_stacking(bit_plane_sliced_images[5], bit_plane_sliced_images[6], bit_plane_sliced_images[7])

        vertical_stacked_images = vertical_stacking(row_1, row_2, row_3)

        img_file_name = f"./Outputs/{image_names[cnt]}_bit_sliced.jpg"
        plt.imsave(img_file_name, vertical_stacked_images, cmap="gray")

        cnt = cnt + 1

    compressed_images = []
    for i in range(0, 3):
        compressed_image_1 = image_compressor(images[i], [7, 6])
        compressed_image_2 = image_compressor(images[i], [7, 6, 5])
        compressed_image_3 = image_compressor(images[i], [7, 6, 5, 4])
        compressed_image_4 = image_compressor(images[i], [7, 6, 5, 4, 3])
        compressed_image_5 = image_compressor(images[i], [7, 6, 5, 4, 3, 2])
        compressed_image_6 = image_compressor(images[i], [7, 6, 5, 4, 3, 2, 1])
        compressed_image_7 = image_compressor(images[i], [7, 6, 5, 4, 3, 2, 1, 0])

        compressed_images.append(compressed_image_1)
        compressed_images.append(compressed_image_2)
        compressed_images.append(compressed_image_3)
        compressed_images.append(compressed_image_4)
        compressed_images.append(compressed_image_5)
        compressed_images.append(compressed_image_6)
        compressed_images.append(compressed_image_7)

        compressed_images = compressed_images[::-1]

        for j in range(7):
            grid = horizontal_stacking(images[i], compressed_images[j])
            img_file_name = f"./Outputs/Compressed/{image_names[i]}_bit_plane_[7-{j}]_plus_bit_sliced.jpg"
            plt.imsave(img_file_name, grid, cmap="gray")
