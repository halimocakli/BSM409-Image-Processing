import cv2
import matplotlib.pyplot as plt
import numpy as np


def horizontal_stacking(*args):
    return np.hstack(args)


def rescale_to_8bits(image):
    s = image.astype(float)
    s = (((s - np.min(s)) / (np.max(s))) * 255).astype(np.uint8)
    return s


def gamma_transformation(r, _c, _gamma):
    r = r.astype(float)
    s = _c * r ** _gamma
    s = rescale_to_8bits(s)
    return s


if __name__ == "__main__":
    c = 1
    gamma_values_1 = [3.0, 4.0, 5.0]
    gamma_values_2 = [0.6, 0.4, 0.3]
    gamma_images = []

    # Dark Images
    img_fractured_spine = cv2.imread("./Images/dark/fractured_spine.tif", 0)
    img_breast_digital_Xray = cv2.imread("./Images/dark/breast_digital_Xray.tif", 0)
    img_DFT_no_log = cv2.imread("./Images/dark/DFT_no_log.tif", 0)

    # Light Images
    img_washed_out_aerial = cv2.imread("./Images/light/washed_out_aerial.tif", 0)
    img_washed_out_pollen = cv2.imread("./Images/light/washed_out_pollen_image.tif")

    for gamma in gamma_values_1:
        gamma_img = gamma_transformation(r=img_washed_out_pollen, _c=c, _gamma=gamma)
        gamma_images.append(gamma_img)

    row_1_pollen = horizontal_stacking(img_washed_out_pollen, gamma_images[0])
    row_2_pollen = horizontal_stacking(*gamma_images[1:])

    grid_pollen = np.vstack((row_1_pollen, row_2_pollen))

    cv2.imshow("POLLEN", grid_pollen)
    plt.imsave("./Outputs/washed_out_pollen_image.jpg", grid_pollen, cmap="gray")

    gamma_images.clear()

    for gamma in gamma_values_2:
        gamma_img = gamma_transformation(r=img_DFT_no_log, _c=c, _gamma=gamma)
        gamma_images.append(gamma_img)

    row_1_DFT = horizontal_stacking(img_DFT_no_log, gamma_images[0])
    row_2_DFT = horizontal_stacking(*gamma_images[1:])

    grid_aerial = np.vstack((row_1_DFT, row_2_DFT))

    cv2.imshow("DFT", grid_aerial)
    plt.imsave("./Outputs/DFT_no_log_gamma_transformed.jpg", grid_aerial, cmap="gray")

    gamma_images.clear()

    for gamma in gamma_values_1:
        gamma_img = gamma_transformation(r=img_washed_out_aerial, _c=c, _gamma=gamma)
        gamma_images.append(gamma_img)

    row_1_aerial = horizontal_stacking(img_washed_out_aerial, gamma_images[0])
    row_2_aerial = horizontal_stacking(*gamma_images[1:])

    grid_aerial = np.vstack((row_1_aerial, row_2_aerial))

    cv2.imshow("AERIAL GAMMA TRANSFORMED", grid_aerial)
    plt.imsave("./Outputs/washed_out_aerial_gamma_transformed.jpg", grid_aerial, cmap="gray")

    gamma_images.clear()

    for gamma in gamma_values_1:
        gamma_img = gamma_transformation(r=img_breast_digital_Xray, _c=c, _gamma=gamma)
        gamma_images.append(gamma_img)

    row_1_breast = horizontal_stacking(img_breast_digital_Xray, gamma_images[0])
    row_2_breast = horizontal_stacking(*gamma_images[1:])

    grid_breast = np.vstack((row_1_breast, row_2_breast))

    cv2.imshow("BREAST DIGITAL XRAY GAMMA TRANSFORMED", grid_breast)
    plt.imsave("./Outputs/breast_digital_Xray_gamma_transformed.jpg", grid_breast, cmap="gray")

    gamma_images.clear()

    for gamma in gamma_values_2:
        gamma_img = gamma_transformation(r=img_fractured_spine, _c=c, _gamma=gamma)
        gamma_images.append(gamma_img)

    row_1_spine = horizontal_stacking(img_fractured_spine, gamma_images[0])
    row_2_spine = horizontal_stacking(*gamma_images[1:])

    grid_fractured_spine = np.vstack((row_1_spine, row_2_spine))

    cv2.imshow("FRACTURED SPINE", grid_fractured_spine)
    cv2.waitKey(0)
    plt.imsave("./Outputs/fractured_spine_gamma_transformed.jpg", grid_fractured_spine, cmap="gray")
