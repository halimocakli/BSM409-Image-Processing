import cv2
import numpy as np
import matplotlib.pyplot as plt
import io

L = 256  # 8 BITS CONSTANT VALUE


def histogram_equalization_custom_function(_image):
    histogram = create_image_histogram_array(_image)
    print(histogram)
    print(histogram.shape)


def create_image_histogram_array(_image):
    histogram, bins = np.histogram(_image, L, range=(0, L))
    return histogram


def create_histogram_visually(_image, _title: str):
    histogram, bins = np.histogram(_image, bins=L, range=(0, L))

    plt.hist(_image.ravel(), bins, [0, L])
    plt.title('Histogram for gray scale picture')
    plt.show()


def from_figure_to_image(fig, dpi=180):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    buffer.seek(0)
    img_arr = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    buffer.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_histogram_image(img):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(img, density=True, range=(0, L))

    histogram_image = from_figure_to_image(fig).mean(axis=2)
    histogram_image = cv2.resize(histogram_image, img.shape)

    fig.clf()
    plt.close(fig)
    return histogram_image


def create_normalized_histogram(img):
    histogram, bins = np.histogram(img, bins=L, range=(0, L))
    return histogram / img.size


def create_cumulative_dist(img):
    normalized_histogram = create_normalized_histogram(img)
    return np.cumsum(normalized_histogram, axis=0)


def equalize_histogram(img):
    shape = img.shape
    ravel = img.ravel()
    cumsum = create_cumulative_dist(ravel)
    values = (L - 1) * cumsum
    eq_img = np.zeros_like(ravel)
    for i, pixel in enumerate(ravel):
        eq_img[i] = values[pixel]
    return eq_img.reshape(shape).astype(np.uint8)


def main_histogram_equalization():
    image_directories = [
        "./Images/top_left.tif",
        "./Images/bottom_left.tif",
        "./Images/2nd_from_top.tif",
        "./Images/third_from_top.tif"
    ]
    original_images = []
    original_histograms = []
    histogram_equalized_images = []
    histogram_equalized_histograms = []

    for image_dir in image_directories:
        image = cv2.imread(image_dir, 0)

        original_images.append(image)
        original_histograms.append(create_histogram_image(image))

        histogram_equalized_image = equalize_histogram(image)
        histogram_equalized_images.append(histogram_equalized_image)

        histogram_equalized_histogram = create_histogram_image(histogram_equalized_image)
        histogram_equalized_histograms.append(histogram_equalized_histogram)

    grid = np.vstack([np.hstack((img, original_hist, hist_eqz_img, hist_eqz_hist))
                      for img, original_hist, hist_eqz_img, hist_eqz_hist in zip(original_images,
                                                                                 original_histograms,
                                                                                 histogram_equalized_images,
                                                                                 histogram_equalized_histograms)])

    plt.imshow(grid, cmap="gray")
    plt.show()
    plt.imsave("./Outputs/output.jpg", grid, cmap="gray")


if __name__ == '__main__':
    main_histogram_equalization()
