
import numpy as np
import cv2


def apply_homography(image, pnt):
    [w1, h1] = [(np.linalg.norm(pnt[0] - pnt[1])),(np.linalg.norm(pnt[1] - pnt[2]))]
    [w2, h2] = [(np.linalg.norm(pnt[2] - pnt[3])),(np.linalg.norm(pnt[3] - pnt[0]))]

    [w, h] = np.int64((np.add((w1, h1), (w2, h2)) / 2)).tolist()

    # find matrix Homography
    dst = np.array([[0, h], [w, h], [w, 0], [0, 0]])
    H = cv2.findHomography(pnt, dst)[0]
    H_inv = np.linalg.inv(H)

    # apply H matrix to image
    res = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
            pnt_src = np.matmul(H_inv, np.array([x, y, 1]))
            pnt_src = np.int32(pnt_src / pnt_src[2])
            res[y, x, :] = image[pnt_src[1], pnt_src[0]]

    return [np.uint8(res), H]


image = cv2.imread('books.jpg')

pnt1 = np.array([[385, 105], [317, 288], [601, 394], [666, 209]])
pnt2 = np.array([[410, 469], [205, 427], [153, 709], [363, 741]])
pnt3 = np.array([[622, 667], [420, 795], [609, 1100], [813, 968]])


[res1, H1] = apply_homography(image, pnt1)
[res2, H2] = apply_homography(image, pnt2)
[res3, H3] = apply_homography(image, pnt3)


print('\nHomography for book1 is : \n', H1)
print('\nHomography for book2 is : \n', H2)
print('\nHomography for book3 is : \n', H3)

cv2.imwrite('res16.jpg', res1)
cv2.imwrite('res17.jpg', res2)
cv2.imwrite('res18.jpg', res3)
