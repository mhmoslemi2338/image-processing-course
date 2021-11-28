import numpy as np
import cv2


def normalize(kernel_in):  # Map to the int(0, 255) range
    kernel = kernel_in.copy()
    kernel = kernel - np.amin(kernel)
    kernel = (kernel / np.amax(kernel)) * 255.0
    return np.uint8(kernel)


 #### preprocess  step ####

img_color = cv2.imread('Greek-ship.jpg')
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (13, 13), 3)

template_color = cv2.imread('patch.png')
template = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)


template = np.int64(cv2.resize(template,
                               (int(0.16 * template.shape[1]),
                                int(0.16 * template.shape[0]))))
img = np.int64(cv2.resize(
    img, (int(0.32 * img.shape[1]), int(0.32 * img.shape[0]))))



#### implement SSD , Zero-mean cross-corr from scratch 

SSD = np.int64(np.zeros((img.shape[0], img.shape[1])))
ZM_ccor = np.int64(np.zeros((img.shape[0], img.shape[1])))

for m in range(0, img.shape[0] - template.shape[0]):
    for n in range(0, img.shape[1] - template.shape[1]):
        window = (img[m:m + template.shape[0], n:n + template.shape[1]])
        ZM_ccor[m, n] = np.sum(np.multiply(
            (window - np.mean(window)), template))
        SSD[m, n] = np.sum(np.power(window - template, 2))


##### apply process on SSD and ZM-cross-corr 

res = (ZM_ccor) - 0.1 * (SSD)
res[res < 0] = 0
res = normalize(res)
_, res = cv2.threshold(res, 110, 255, cv2.THRESH_BINARY)
res = cv2.resize(
    res, (int(1 / 0.32 * res.shape[1]), int(1 / 0.32 * res.shape[0])))


#### find cordinate of best find results 

pnt = []
for j in range(0, res.shape[1]):
    tmp = (np.where(res[:, j] > 0))
    if tmp[0].tolist() != []:
        pnt.append([j, int(np.average(tmp))])

pnt_final = [pnt[0]]
for i, row in enumerate(pnt):
    last = np.array(pnt_final[-1])
    dist = np.linalg.norm(last - np.array(row))
    if dist > 30:
        if np.abs(last[0] - row[0]) > 20:
            pnt_final.append(row)


### draw results on the original image 

img_color = cv2.imread('Greek-ship.jpg')
template_color = cv2.imread('patch.png')
(h, w) = template_color.shape[:-1]

img_res = img_color.copy()
for y, x in pnt_final:
    img_res = cv2.rectangle(img_res, (y, x), (y + w, x + h), 255, 5)

cv2.imwrite('res15.jpg', img_res)
