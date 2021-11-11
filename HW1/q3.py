import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import os


def find_crop(image):
    h = image.shape[0]
    avg = 0
    for i in range(-50, 50):
        avg += np.average(image[int(h / 2) + i, :])
    avg = avg / 100
    up, down, left, right = 0, 0, 0, 0
    for i in range(300):
        if np.average(image[-i, :]) / avg < 0.6 or np.average(image[-i, :]) / avg > 1.4:
            down = i
        if np.average(image[i, :]) / avg < 0.6 or np.average(image[i, :]) / avg > 1.4:
            up = i
        if np.average(image[:, i]) /  avg < 0.6 or np.average(image[:, i]) / avg > 1.4:
            left = i
        if np.average(image[:, -i]) / avg < 0.6 or np.average(image[:, -i]) / avg > 1.4:
            right = i
    return [up, down, left, right]


def find_shift(in1, in2):  # im1: fixed , im2:moving
    (h, w) = in1.shape
    img1 = in1.copy()
    img2 = in2.copy()
    img1 = img1[int(0.35 * h):int(0.65 * h), int(0.25 * w):int(0.75 * w)]
    img2 = img2[int(0.35 * h):int(0.65 * h), int(0.25 * w):int(0.75 * w)]

    scale = 0.1
    (h, w) = np.shape(img1)
    dim = (int(scale * w), int(scale * h))
    img1_ = np.int64(cv2.resize(img1, dim))
    img2_ = np.int64(cv2.resize(img2, dim))
    distances = []
    for x in range(-10, 10):
        for y in range(-10, 10):
            tmp = scipy.ndimage.shift(img2_, (y, x))
            l1 = np.sum(np.abs(tmp - img1_))
            distances.append([l1, (y, x)])
    distances.sort(key=lambda x: x[0])
    shift = tuple(np.array(distances[0][1]) * 5)

    scale = 0.5
    dim = (int(scale * w), int(scale * h))
    img1_ = np.int64(cv2.resize(img1, dim))
    img2_ = np.int64(cv2.resize(img2, dim))
    distances.clear()
    for x in range(-2 + shift[1], 3 + shift[1]):
        for y in range(-2 + shift[0], 3 + shift[0]):
            tmp = scipy.ndimage.shift(img2_, (y, x))
            l1 = np.sum(np.abs(tmp - img1_))
            distances.append([l1, (y, x)])
    distances.sort(key=lambda x: x[0])
    shift = tuple(np.array(distances[0][1]) * 2)

    distances.clear()
    for x in range(-1 + shift[1], 2 + shift[1]):
        for y in range(-1 + shift[0], 2 + shift[0]):
            tmp = np.int64(scipy.ndimage.shift(img2, (y, x)))
            l1 = np.sum(np.abs(tmp - np.int64(img1)))
            distances.append([l1, (y, x)])
    distances.sort(key=lambda x: x[0])
    s1 = tuple(np.array(distances[0][1]))
    return s1


def main(name_in, name_out):
    img = cv2.imread(name_in, cv2.IMREAD_UNCHANGED)

    (h, w) = np.shape(img)
    h_ = int(h / 3)
    im1 = img[0:h_, :]
    im2 = img[h_:2 * h_, :]
    im3 = img[2 * h_:3 * h_, :]


    s21= find_shift(im2, im1)
    s23= find_shift(im2, im3)

    im1 = scipy.ndimage.shift(im1, s21)
    im3 = scipy.ndimage.shift(im3, s23)

    tmp = np.array([find_crop(im1), find_crop(im2), find_crop(im3)])
    [up, down, left, right] = np.max(tmp, axis=0)

    im1 = im1[up:-down, left:-right]
    im2 = im2[up:-down, left:-right]
    im3 = im3[up:-down, left:-right]

    out = cv2.merge([im1, im2, im3])
    out = (out / 256).astype('uint8')

    cv2.imwrite(name_out, out)
    print('for image '+name_in)
    print( 'shift for blue to fit on green : ',s21)
    print( 'shift for red to fit on green : ',s23)
    print("\n")


main('master-pnp-prok-01800-01886a.tif', 'res03-Amir.jpg')
main('master-pnp-prok-01800-01833a.tif', 'res04-Mosque.jpg')
main('master-pnp-prok-00400-00458a.tif', 'res05-Train.jpg')
