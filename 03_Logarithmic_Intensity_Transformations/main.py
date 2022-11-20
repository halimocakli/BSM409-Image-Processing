import cv2
import matplotlib.pyplot as plt
import numpy as np


def logarithmic_transformer(r, c):
    """
    :param r: Üzerinde logaritma dönüşümü yapılacak giriş imajı.
    :param c: Sabit sayı.
    :return s: Logaritma dönüşümü uygulanmış çıktı imajı.
    """
    r = r.astype(float)
    s = c * np.log(1 + r)
    s = rescale_to_8bits(s)
    return s.astype(np.uint8)


def rescale_to_8bits(image):
    s = image.astype(float)
    s = (((s - np.min(s)) / (np.max(s))) * 255).astype(np.uint8)
    return s


def horizontal_stacking(*args):
    return np.hstack(args)


if __name__ == '__main__':
    image_DFT = cv2.imread("./Images/DFT_no_log.tif", 0)
    image_DFT_log_transformed = logarithmic_transformer(r=image_DFT, c=1)

    images_adjoined = horizontal_stacking(image_DFT, image_DFT_log_transformed)

    cv2.imshow("DFT AND LOGARITHMIC TRANSFORMED DFT IMAGES", images_adjoined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.imsave("./Outputs/image_DFT_log_transformed_scaled.jpg", image_DFT_log_transformed, cmap="gray")
    plt.imsave("./Outputs/DFT_and_image_DFT_log_transformed_scaled_adjoined.jpg", images_adjoined, cmap="gray")
