
import cv2
import numpy as np


img = cv2.imread('Enhance2.JPG')

a = 0.075

img_b = np.array(img[:, :, 0], dtype=np.float64)
img_g = np.array(img[:, :, 1], dtype=np.float64)
img_r = np.array(img[:, :, 2], dtype=np.float64)

img_b2 = np.uint8(np.log10(a * img_b + 1) * (255 / np.log10(1 + 255 * a)))
img_g2 = np.uint8(np.log10(a * img_g + 1) * (255 / np.log10(1 + 255 * a)))
img_r2 = np.uint8(np.log10(a * img_r + 1) * (255 / np.log10(1 + 255 * a)))

img2 = img.copy()
img2[:, :, 0] = img_b2
img2[:, :, 1] = img_g2
img2[:, :, 2] = img_r2


cv2.imwrite('res02.jpg', img2)
