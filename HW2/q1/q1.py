import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal, ndimage


def filter_generate(img, D):
    (h, w) = img.shape[0:2]
    LPF = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            tmp = (i - h // 2)**2 + (j - w // 2)**2
            LPF[i, j] = np.exp(-tmp / (2 * D**2))
    HPF = 1 - LPF
    return [HPF, LPF]


def normalize(kernel_in):  # Map to the int(0, 255) range
    kernel = kernel_in.copy()
    kernel = kernel - np.amin(kernel)
    kernel = (kernel / np.amax(kernel)) * 255.0
    return np.uint8(kernel)


#****************************#
#********** part 1 **********#
#****************************#

img = np.float64(cv2.imread('flowers.blur.png'))

k = 11
gaussian = cv2.getGaussianKernel(k, 3)
gaussian = np.outer(gaussian, gaussian)


img_gauss = img.copy()
img_gauss[:, :, 0] = signal.convolve2d(img[:, :, 0], gaussian, 'same')
img_gauss[:, :, 1] = signal.convolve2d(img[:, :, 1], gaussian, 'same')
img_gauss[:, :, 2] = signal.convolve2d(img[:, :, 2], gaussian, 'same')


unsharp_mask = img - img_gauss
img_sharp1 = img + 0.5 * (unsharp_mask)


gaussian = cv2.resize(gaussian,
                      (int((600 / k) * gaussian.shape[1]),
                       int((600 / k) * gaussian.shape[0])))
cv2.imwrite('res01.jpg', normalize(gaussian))
cv2.imwrite('res02.jpg', normalize(img_gauss))
cv2.imwrite('res03.jpg', normalize(unsharp_mask))
cv2.imwrite('res04.jpg', normalize(img_sharp1))


#****************************#
#********** part 2 **********#
#****************************#

img = np.float64(cv2.imread('flowers.blur.png'))


k = 11
gauss_kernel = cv2.getGaussianKernel(k, 2.5)
gauss_kernel = np.outer(gauss_kernel, gauss_kernel)
LOG = ndimage.laplace(gauss_kernel)


img_LOG = img.copy()
img_LOG[:, :, 0] = signal.convolve2d(img[:, :, 0], LOG, 'same')
img_LOG[:, :, 1] = signal.convolve2d(img[:, :, 1], LOG, 'same')
img_LOG[:, :, 2] = signal.convolve2d(img[:, :, 2], LOG, 'same')


img_sharp2 = img - 1.8 * (img_LOG)


LOG = cv2.resize(LOG,
                 (int((600 / k) * LOG.shape[1]),
                  int((600 / k) * LOG.shape[0])))
cv2.imwrite('res05.jpg', normalize(LOG))
cv2.imwrite('res06.jpg', normalize(img_LOG))
cv2.imwrite('res07.jpg', normalize(img_sharp2))


#****************************#
#********** part 3 **********#
#****************************#
img = cv2.imread('flowers.blur.png')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hsv[:, :, 2] = img_hsv[:, :, 2].astype(np.float64)
value = img_hsv[:, :, 2]

[HPF, _] = filter_generate(img, 80)


value_FFT = np.fft.fftshift(np.fft.fft2(value))
filtered_value_FFT = np.multiply(value_FFT, (0.2 * HPF + 1))

img_hsv[:, :, 2] = normalize(
    np.real(np.fft.ifft2(np.fft.ifftshift(filtered_value_FFT))))
img_hsv[:, :, 2] = img_hsv[:, :, 2].astype(np.uint8)


img_sharp3 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


cv2.imwrite('res08.jpg', normalize(np.log10(np.abs(value_FFT) + 0.001)))
cv2.imwrite('res09.jpg', normalize(HPF))
cv2.imwrite(
    'res10.jpg',
    normalize(
        np.log10(
            np.abs(filtered_value_FFT) +
            0.001)))
cv2.imwrite('res11.jpg', img_sharp3)


#****************************#
#********** part 4 **********#
#****************************#
img = cv2.imread('flowers.blur.png')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hsv[:, :, 2] = img_hsv[:, :, 2]
value = np.float64(img_hsv[:, :, 2])


value_FFT = np.fft.fftshift(np.fft.fft2(value))


(c1, c2) = (img.shape[0] // 2, img.shape[1] // 2)
coeff = np.zeros(value_FFT.shape)
for i in range(c1):
    for j in range(c2):
        x = (i**2 + j**2)
        coeff[c1 + i, c2 + j] = x
        coeff[c1 + i, c2 - j] = x
        coeff[c1 - i, c2 + j] = x
        coeff[c1 - i, c2 - j] = x
value_FFT_filtered = np.multiply(4 * coeff * ((np.pi)**2), value_FFT)


value_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(value_FFT_filtered)))
img_hsv[:, :, 2] = normalize((value + 0.0000005 * value_filtered))
img_sharp4 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


cv2.imwrite(
    'res12.jpg',
    normalize(
        np.log10(
            np.abs(value_FFT_filtered) +
            0.001)))
cv2.imwrite('res13.jpg', normalize(value_filtered))
cv2.imwrite('res14.jpg', img_sharp4)
