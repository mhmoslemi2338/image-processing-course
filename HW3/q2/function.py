import numpy as np
import cv2


def random_patch(img, patch, block_size, overlap, mask_bool=False, mask=0 ):
    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    (h, w, _) = img.shape
    img_gray = cv2.cvtColor(
        img[:h - (block_size - overlap), :w - (block_size - overlap), :], cv2.COLOR_BGR2GRAY)
    if not mask_bool:
        res = cv2.matchTemplate(img_gray,patch_gray,cv2.TM_CCOEFF_NORMED)  # normalized cross-cor
        tmp = (np.argsort(res.reshape(-1)))

    else:
        res = cv2.matchTemplate(img_gray,patch_gray,cv2.TM_SQDIFF,mask=mask)  # normalized cross-cor
        tmp = np.flip(np.argsort(res.reshape(-1)))


    top_left = np.flip((np.unravel_index(tmp, res.shape))).T
    pre = np.array([0, 0])
    good = []
    for i, row in enumerate(top_left[0:500]):
        if i > 1:
            tmp1 = np.min(np.abs(np.array(good) - pre), axis=0)
            tmp = np.min(np.sum(np.power(np.array(good) - row, 2), axis=1))
            pre = row
            if tmp < 30 or tmp1[0] < 3 or tmp1[0] < 3:
                continue
        pre = row
        good.append(row)
    good = np.array(good)
    np.random.shuffle(good)
    top_left = (good[0]).tolist()
    patch2 = img[top_left[1]:top_left[1] + block_size,top_left[0]:top_left[0] + block_size, :]
    return patch2

def min_cut_path(margin1_gray, margin2_gray):
    errors = np.power(margin1_gray - margin2_gray, 2)
    cost = np.zeros(errors.shape)
    cost[-1] = errors[-1]
    path_matrix = np.zeros(errors.shape)

    errors = np.pad(errors, [(0, 0), (1, 1)],mode='constant', constant_values=np.inf)
    cost = np.pad(cost, [(0, 0), (1, 1)],mode='constant', constant_values=np.inf)

    for i in range(len(errors) - 2, -1, -1):
        for j in range(1, len(errors[i]) - 1):
            cost[i][j] = errors[i][j] + min(cost[i + 1][j - 1:j + 2])
            path_matrix[i][j -1] = np.argmin(cost[i + 1][j - 1:j + 2]) - 1 + j - 1

    path = [int(path_matrix[0][np.argmin(cost[0]) - 1])]
    for i in range(1, len(path_matrix)):
        path.append(int(path_matrix[i][path[-1]]))
    return path


def combine_margin(margin1_, margin2_, axis='h'):
    margin1 = margin1_.copy()
    margin2 = margin2_.copy()

    if axis == 'v':
        margin1 = cv2.merge([margin1[:, :, 0].T, margin1[:, :, 1].T, margin1[:, :, 2].T])
        margin2 = cv2.merge([margin2[:, :, 0].T, margin2[:, :, 1].T, margin2[:, :, 2].T])

    margin1_gray = np.float64(cv2.cvtColor(margin1, cv2.COLOR_BGR2GRAY))
    margin2_gray = np.int64(cv2.cvtColor(margin2, cv2.COLOR_BGR2GRAY))
    path = min_cut_path(margin1_gray, margin2_gray)
    patch_combined = np.zeros(margin1.shape)

    for i, row in enumerate(path):
        patch_combined[i] = margin2[i]
        patch_combined[i, :row] = margin1[i, :row]
    if axis == 'v':
        a = patch_combined[:, :, 0]
        b = patch_combined[:, :, 1]
        c = patch_combined[:, :, 2]
        patch_combined = cv2.merge([a.T, b.T, c.T])
    return np.uint8(patch_combined)
