import cv2
import matplotlib.pyplot as plt
import numpy as np


def stack(*args):
    return np.hstack(args)


def rescale(img):
    s = img.astype(float)
    s = s - np.min(s)
    s = s / np.max(s)
    return (s * 255).astype(np.uint8)


def img_negative(img):
    L = np.max(img)
    negative_image = L - img
    return negative_image


def log_transformation(r, c):
    r = r.astype(float)
    s = c * np.log(1 + r)
    s = rescale(s)
    return s


def gamma_transformation(r, c, gamma):
    r = r.astype(float)
    s = c * r ** gamma
    s = rescale(s)
    return s


c = 1
gamma_values_1 = [3.0, 4.0, 5.0]
gamma_values_2 = [0.6, 0.4, 0.3]
gamma_images = []

img_fractured_spine = cv2.imread("./images/fractured_spine.tif", 0)
img_breast_digital_Xray = cv2.imread("./images/breast_digital_Xray.tif", 0)
img_washed_out_aerial = cv2.imread("./images/washed_out_aerial.tif", 0)
img_DFT_no_log = cv2.imread("./images/DFT_no_log.tif", 0)

for gamma in gamma_values_2:
    gamma_img = gamma_transformation(r=img_DFT_no_log, c=c, gamma=gamma)
    gamma_images.append(gamma_img)

row_1_DFT = stack(img_DFT_no_log, gamma_images[0])
row_2_DFT = stack(*gamma_images[1:])

grid_aerial = np.vstack((row_1_DFT, row_2_DFT))

plt.imshow(grid_aerial, cmap="gray")
plt.show()
plt.imsave("DFT_no_log_gamma_transformed.jpg", grid_aerial, cmap="gray")

gamma_images = []

for gamma in gamma_values_1:
    gamma_img = gamma_transformation(r=img_washed_out_aerial, c=c, gamma=gamma)
    gamma_images.append(gamma_img)

row_1_aerial = stack(img_washed_out_aerial, gamma_images[0])
row_2_aerial = stack(*gamma_images[1:])

grid_aerial = np.vstack((row_1_aerial, row_2_aerial))

plt.imshow(grid_aerial, cmap="gray")
plt.show()
plt.imsave("washed_out_aerial_gamma_transformed.jpg", grid_aerial, cmap="gray")

gamma_images = []

for gamma in gamma_values_1:
    gamma_img = gamma_transformation(r=img_breast_digital_Xray, c=c, gamma=gamma)
    gamma_images.append(gamma_img)

row_1_breast = stack(img_breast_digital_Xray, gamma_images[0])
row_2_breast = stack(*gamma_images[1:])

grid_breast = np.vstack((row_1_breast, row_2_breast))

plt.imshow(grid_breast, cmap="gray")
plt.imsave("breast_digital_Xray_gamma_transformed.jpg", grid_breast, cmap="gray")
plt.show()

gamma_images = []

for gamma in gamma_values_2:
    gamma_img = gamma_transformation(r=img_fractured_spine, c=c, gamma=gamma)
    gamma_images.append(gamma_img)

row_1_spine = stack(img_fractured_spine, gamma_images[0])
row_2_spine = stack(*gamma_images[1:])

grid_fractured_spine = np.vstack((row_1_spine, row_2_spine))

plt.imshow(grid_fractured_spine, cmap="gray")
plt.imshow()
plt.imsave("fractured_spine_gamma_transformed.jpg", grid_fractured_spine, cmap="gray")
