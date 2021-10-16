
import cv2
import numpy as np
import matplotlib.pyplot as plt


pink = cv2.imread('Pink.JPG')
pink = cv2.cvtColor(pink, cv2.COLOR_BGR2HSV)
pink_value = pink[:, :, 2].flatten()  # get brigtness channel
y, x = np.histogram(pink_value, bins=np.arange(256), density=True)
H2 = np.cumsum(y)
H2 = np.uint8(255 * np.float64(H2 / np.sum(y)))


dark = cv2.imread('Dark.jpg')
dark = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)
dark_value = dark[:, :, 2].flatten()  # get brigtness channel
y, x = np.histogram(dark_value, bins=np.arange(256), density=True)
H1 = np.cumsum(y)
H1 = np.uint8(255 * np.float64(H1 / np.sum(y)))


tmp1 = np.interp(dark_value.flatten(), x[:-1], H1)
value = np.interp(tmp1.flatten(), H2, x[:-1])

dark[:, :, 2] = value.reshape(np.shape(dark)[0:2])
dark = cv2.cvtColor(dark, cv2.COLOR_HSV2BGR)
cv2.imwrite('res11.jpg', dark)


y, x = np.histogram(value, bins=np.arange(256), density=True)
y = y / np.sum(y)

fig = plt.figure(figsize=(12, 12))
plt.plot(x[:-1], y)
plt.title('hsitogram for Dark.jpg after hist specificatin')
plt.grid()
plt.ylabel('normalized value', fontsize='xx-large')
plt.xlabel('Pixel value', fontsize='xx-large')
fig.savefig('res10.jpg', dpi=3 * fig.dpi)
plt.close(fig)
