import cv2
import numpy as np
import timeit
import scipy.signal


def shift(image, tx, ty):
    (h, w, _) = np.shape(image)
    T = np.array([[1, 0, tx],
                  [0, 1, ty]], dtype=np.float32)
    image = cv2.warpAffine(image, T, (w, h))
    return image


############ method one ####################
img = cv2.imread('Pink.jpg')
start = timeit.default_timer()


kernel=np.ones((3,3))
res=img[1:-1,1:-1,:].copy()
res[:,:,0]=(scipy.signal.convolve2d(img[:,:,0],kernel,mode='valid'))/9
res[:,:,1]=(scipy.signal.convolve2d(img[:,:,1],kernel,mode='valid'))/9
res[:,:,2]=(scipy.signal.convolve2d(img[:,:,2],kernel,mode='valid'))/9
res=np.uint8(res)

stop = timeit.default_timer()
print('Run-Time for method one : %0.4f sec' % (stop - start))
cv2.imwrite('res07.jpg', res)


############ method two ####################
img = cv2.imread('Pink.jpg')
img2 = img.copy()
start = timeit.default_timer()

(h, w, _) = np.shape(img2)
for i in range(h - 2):
    for j in range(w - 2):
        img2[i + 1, j + 1, 0] = \
            np.uint8(np.float64(np.sum(img[i:i + 3, j:j + 3, 0])) / 9)
        img2[i + 1, j + 1, 1] = \
            np.uint8(np.float64(np.sum(img[i:i + 3, j:j + 3, 1])) / 9)
        img2[i + 1, j + 1, 2] = \
            np.uint8(np.float64(np.sum(img[i:i + 3, j:j + 3, 2])) / 9)
res = img2[1:-1, 1:-1, :]

stop = timeit.default_timer()
print('Run-Time for method two : %0.3f sec' % (stop - start))
cv2.imwrite('res08.jpg', res)


############ method Three ####################
img = cv2.imread('Pink.jpg')
start = timeit.default_timer()

img2 = np.float64(img.copy())  # base
img2 += np.float64(shift(img, 1, 0))  # left
img2 += np.float64(shift(img, -1, 0))  # right
img2 += np.float64(shift(img, 0, -1))  # up
img2 += np.float64(shift(img, 0, 1))  # down
img2 += np.float64(shift(img, 1, 1))  # down + left
img2 += np.float64(shift(img, -1, 1))  # down + right
img2 += np.float64(shift(img, 1, -1))  # up + left
img2 += np.float64(shift(img, -1, -1))  # up + right
img2 = np.uint8((img2) / 9)
res = img2[1:-1, 1:-1, :]

stop = timeit.default_timer()
print('Run-Time for method Three : %0.3f sec' % (stop - start))
cv2.imwrite('res09.jpg', res)
