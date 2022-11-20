import cv2
import numpy as np

image_strawberries = cv2.imread("./Images/strawberries.png")
image_vegetables = cv2.imread("./Images/vegetables.png")
image_baboon = cv2.imread("./Images/baboon.png")

print(f"Shape of strawberries image: {image_strawberries.shape}")
print(f"Shape of vegetables image: {image_vegetables.shape}")
print(f"Shape of baboon image: {image_baboon.shape}")

print("--------------------------------------------------------")

print(f"Maximum value of strawberries image: {np.max(image_strawberries)}")
print(f"Minimum value of strawberries image: {np.min(image_strawberries)}")

print("--------------------------------------------------------")

print(f"Maximum value of vegetables image: {np.max(image_vegetables)}")
print(f"Minimum value of vegetables image: {np.min(image_vegetables)}")

print("--------------------------------------------------------")

print(f"Maximum value of baboon image: {np.max(image_baboon)}")
print(f"Minimum value of baboon image: {np.min(image_baboon)}")

print("--------------------------------------------------------")

print(f"Strawberries image shape of BLUE CHANNEL: {image_strawberries[:, :, 0].shape}")
print(f"Strawberries image shape of GREEN CHANNEL: {image_strawberries[:, :, 1].shape}")
print(f"Strawberries image shape of RED CHANNEL: {image_strawberries[:, :, 2].shape}")

print("--------------------------------------------------------")

print(f"Vegetables image shape of BLUE CHANNEL: {image_vegetables[:, :, 0].shape}")
print(f"Vegetables image shape of GREEN CHANNEL: {image_vegetables[:, :, 1].shape}")
print(f"Vegetables image shape of RED CHANNEL: {image_vegetables[:, :, 2].shape}")

print("--------------------------------------------------------")

print(f"Baboon image shape of BLUE CHANNEL: {image_baboon[:, :, 0].shape}")
print(f"Baboon image shape of GREEN CHANNEL: {image_baboon[:, :, 1].shape}")
print(f"Baboon image shape of RED CHANNEL: {image_baboon[:, :, 2].shape}")

print("--------------------------------------------------------")

x = 125
y = 25

blue_channel = 0
green_channel = 1
red_channel = 2

intensity_blue_strawberries = image_strawberries[y, x, blue_channel]
intensity_green_strawberries = image_strawberries[y, x, green_channel]
intensity_red_strawberries = image_strawberries[y, x, red_channel]

intensity_blue_vegetables = image_vegetables[y, x, blue_channel]
intensity_green_vegetables = image_vegetables[y, x, green_channel]
intensity_red_vegetables = image_vegetables[y, x, red_channel]

intensity_blue_baboon = image_baboon[y, x, blue_channel]
intensity_green_baboon = image_baboon[y, x, green_channel]
intensity_red_baboon = image_baboon[y, x, red_channel]

print(f"Intensity of Strawberries Image at [Y:{y} X:{x} BLUE CHANNEL:{blue_channel}] = {intensity_blue_strawberries}")
print(
    f"Intensity of Strawberries Image at [Y:{y} X:{x} GREEN CHANNEL:{green_channel}] = {intensity_green_strawberries}")
print(f"Intensity of Strawberries Image at [Y:{y} X:{x} RED CHANNEL:{red_channel}] = {intensity_red_strawberries}")

print("--------------------------------------------------------")

print(f"Intensity of Vegetables Image at [Y:{y} X:{x} BLUE CHANNEL:{blue_channel}] = {intensity_blue_vegetables}")
print(f"Intensity of Vegetables Image at [Y:{y} X:{x} GREEN CHANNEL:{green_channel}] = {intensity_green_vegetables}")
print(f"Intensity of Vegetables Image at [Y:{y} X:{x} RED CHANNEL:{red_channel}] = {intensity_red_vegetables}")

print("--------------------------------------------------------")

print(f"Intensity of Baboon Image at [Y:{y} X:{x} BLUE CHANNEL:{blue_channel}] = {intensity_blue_baboon}")
print(f"Intensity of Baboon Image at [Y:{y} X:{x} GREEN CHANNEL:{green_channel}] = {intensity_green_baboon}")
print(f"Intensity of Baboon Image at [Y:{y} X:{x} RED CHANNEL:{red_channel}] = {intensity_red_baboon}")

print("--------------------------------------------------------")

print(f"Intensity of Strawberries Image at [Y:{y} X:{x}] = {image_strawberries[y, x]}")
print(f"Intensity of Vegetables Image at [Y:{y} X:{x}]: {image_vegetables[y, x]}")
print(f"Intensity of Baboon Image at [Y:{y} X:{x}] = {image_baboon[y, x]}")

crop_strawberries = image_strawberries[25:125, 50:150]
crop_vegetables = image_vegetables[125:200, 90:255]
crop_baboon = image_baboon[10:160, 0:]

cv2.imshow("Strawberries", image_strawberries)
cv2.imshow("Vegetables", image_vegetables)
cv2.imshow("Baboon", image_baboon)

cv2.imshow("CROP Strawberries", crop_strawberries)
cv2.imshow("CROP Vegetables", crop_vegetables)
cv2.imshow("CROP Baboon", crop_baboon)

strawberries_image_RGB_adjoined = np.hstack(
    [image_strawberries[:, :, 0], image_strawberries[:, :, 1], image_strawberries[:, :, 2]])
vegetables_image_RGB_adjoined = np.hstack(
    [image_vegetables[:, :, 0], image_vegetables[:, :, 1], image_vegetables[:, :, 2]])
baboon_image_RGB_adjoined = np.hstack([image_baboon[:, :, 0], image_baboon[:, :, 1], image_baboon[:, :, 2]])

cv2.imshow("BLUE - GREEN - RED CHANNELS OF STRAWBERRIES IMAGE", strawberries_image_RGB_adjoined)
cv2.imshow("BLUE - GREEN - RED CHANNELS OF VEGETABLES IMAGE", vegetables_image_RGB_adjoined)
cv2.imshow("BLUE - GREEN - RED CHANNELS OF BABOON IMAGE", baboon_image_RGB_adjoined)

cv2.waitKey(0)
