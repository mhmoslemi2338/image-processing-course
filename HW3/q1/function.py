

import cv2
import numpy as np
from scipy.signal import argrelextrema


def normalize(kernel_in):  # Map to the int(0, 255) range
    kernel = kernel_in.copy()
    kernel = kernel - np.amin(kernel)
    kernel = (kernel / np.amax(kernel)) * 255.0
    return np.uint8(kernel)


def hough_space(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, (np.min(gray.shape), np.min(gray.shape)))
    edge = cv2.Canny(gray, threshold1=70, threshold2=110)
    edge_h, edge_w = edge.shape
    d = int(np.sqrt(edge_h**2 + edge_w**2))
    thetas = np.linspace(-np.pi, np.pi, 360)
    rhos = np.linspace(-d, d, 2 * d)
    accumulator = np.zeros([len(rhos), len(thetas)])
    for y in range(edge_h):
        for x in range(edge_w):
            if edge[y, x] != 255:
                continue
            r = x * np.cos(thetas) + y * np.sin(thetas)
            for i, row in enumerate(r):
                r_index = np.argmin(np.abs(rhos - row))
                accumulator[r_index][i] += 1
    return [accumulator, edge]


def find_lines(img, accumulator, thresh=150):
    h = np.min(img.shape[0:2])
    d = int(np.sqrt(h**2 + h**2))
    thetas = np.linspace(-np.pi, np.pi, 360)
    rhos = np.linspace(-d, d, 2 * d)
    res = argrelextrema(accumulator.reshape(-1), np.greater)
    (X, Y) = np.unravel_index(res[0], accumulator.shape)
    lines = []
    for i, x in enumerate(X):
        y = Y[i]
        if accumulator[x][y] < thresh:
            continue
        rho = rhos[x]
        theta = thetas[y]
        lines.append([theta, rho])
    return lines


def make_mb(my_line):
    lines = []
    for row in my_line:
        [theta, rho] = [row[0], row[1]]
        if np.abs(np.abs(theta) - np.pi / 2) < np.deg2rad(0.2):
            lines.append([rho, np.inf, theta, rho])
        elif np.abs(np.abs(theta) - np.pi) < np.deg2rad(0.2):
            lines.append([rho, -np.inf, theta, rho])
        else:
            b = rho / np.sin(theta)
            m = -np.cos(theta) / np.sin(theta)
            lines.append([m, b, theta, rho])
    return lines


def avg_near_line(lines_in, th1=0.036, th2=0.15):
    lines = lines_in.copy()
    lines.sort(key=lambda x: x[1])
    tmp = [lines[0]]
    flag = tmp[-1]
    lines_out = []
    for i in range(1, len(lines)):
        [theta, rho] = lines[i]
        if abs(rho - flag[1]) / abs(flag[1]) < th1:
            if abs(theta - flag[0]) / abs(flag[0]) < th2:
                tmp.append(lines[i])
                flag = (np.mean(tmp, axis=0)).tolist()
                continue
        lines_out.append(np.mean(tmp, axis=0).tolist())
        tmp = [lines[i]]
        flag = tmp[-1]
    lines_out.append(np.mean(tmp, axis=0).tolist())
    return lines_out


def select_parallel_lines(in_lines, th=2):
    in_lines.sort(key=lambda x: x[0])
    tmp = [in_lines[0]]
    flag = tmp[-1]
    lines_out = []
    for i in range(1, len(in_lines)):
        [theta, _] = in_lines[i]
        if abs(theta - flag[0]) / abs(flag[0]) < 0.02:
            tmp.append(in_lines[i])
            flag = (np.mean(tmp, axis=0)).tolist()
            continue
        if len(tmp) > th:
            for row in tmp:
                lines_out.append(row)
        tmp = [in_lines[i]]
        flag = in_lines[i]
    if len(tmp) > th:
        for row in tmp:
            lines_out.append(row)
    return lines_out


def find_chess_lines(img, lines_in):
    linef = []
    lines_in_mb = make_mb(lines_in)
    for k, row in enumerate(lines_in_mb):
        mask_up = np.uint8(np.zeros(img.shape))
        mask_down = np.uint8(np.zeros(img.shape))
        [m, b] = [row[0], row[1]]
        x = np.arange(int(img.shape[1] * 0.4), int(img.shape[1] * 0.9))
        y = (np.int64(m * x + b)).tolist()
        x = x.tolist()
        for i in range(len(x)):
            if not int(img.shape[1] * 0.4) < y[i] < int(img.shape[1] * 0.8):
                continue
            try:
                margin = 35
                if margin < x[i] < img.shape[0] - margin and margin < y[i] < img.shape[0] - margin:
                    mask_down[y[i] - margin:y[i], x[i] - margin:x[i],:] = img[y[i] - margin:y[i], x[i] - margin:x[i], :]
                    mask_up[y[i] - margin:y[i], x[i] - margin:x[i],:] = img[y[i]:y[i] + margin, x[i]:x[i] + margin, :]
            except BaseException:
                pass
        mask_down = cv2.cvtColor(mask_down, cv2.COLOR_BGR2GRAY)
        mask_up = cv2.cvtColor(mask_up, cv2.COLOR_BGR2GRAY)
        res1 = cv2.matchTemplate(mask_down, mask_up, cv2.TM_CCOEFF_NORMED)
        res2 = cv2.matchTemplate(mask_up, mask_down, cv2.TM_CCOEFF_NORMED)
        res = (res1[0][0] + res2[0][0]) / 2
        if 0.53 < res < 0.78:
            linef.append(lines_in[k])
    return linef


def draw_line(im, lines, thick=2):
    img = cv2.resize(im.copy(), (np.min(im.shape[0:2]), np.min(im.shape[0:2])))
    for row in lines:
        [m, b] = [row[0], row[1]]
        if b == np.inf:
            cv2.line(img, (-5000, round(m)), (5000, round(m)),(0, 0, 255), thickness=thick)
        elif b == -np.inf:
            cv2.line(img, (-round(m), -5000), (-round(m), 5000),(0, 0, 255), thickness=thick)
        else:
            p1 = (-4000, int(-m * 4000 + b))
            p2 = (4000, int(m * 4000 + b))
            cv2.line(img, p1, p2, (0, 0, 255), thickness=thick)
    return img


def intersection(img, line_in):
    linef2_mb = make_mb(line_in)
    points = []
    for i in range(len(linef2_mb)):
        for j in range(i + 1, len(linef2_mb)):
            [m1, b1] = [linef2_mb[i][0], linef2_mb[i][1]]
            [m2, b2] = [linef2_mb[j][0], linef2_mb[j][1]]
            if abs(m2 - m1) / abs(m1) > 0.3:
                xi = (b1 - b2) / (m2 - m1)
                yi = m1 * xi + b1
                if 0 < xi < img.shape[0] and 0 < yi < img.shape[0]:
                    points.append([int(xi), int(yi)])
    return np.int64(avg_near_line(points, 0.04, 0.04))
