
import numpy as np
import cv2
import matplotlib.pyplot as plt


def filter_generate(img, D):
    (h, w) = img.shape[0:2]
    LPF = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            tmp = (i - h // 2)**2 + (j - w // 2)**2
            LPF[i, j] = np.exp(-tmp / (2 * D**2))
    HPF = 1 - LPF
    return [HPF, LPF]


def my_save(img, name):
    fig = plt.figure()
    plt.subplot(131)
    plt.imshow(np.log10(abs(img[0]) + 0.0001), 'gray')
    plt.title('channel Blue')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(np.log10(abs(img[1]) + 0.0001), 'gray')
    plt.title('channel Green')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(np.log10(abs(img[2]) + 0.0001), 'gray')
    plt.title('channel Red')
    plt.axis('off')
    fig.savefig(name, dpi=4 * fig.dpi)
    plt.close(fig)


near = cv2.imread('res19-near.jpg')
far = cv2.imread('res20-far.jpg')


eye_far = np.array([[143, 208], [224, 208]])
eye_near = np.array([[150, 219], [250, 219]])

scale = np.linalg.norm(eye_near[0] - eye_near[1]) / \
    np.linalg.norm(eye_far[0] - eye_far[1])
(h, w) = far.shape[:-1]
dim = (int(scale * w), int(scale * h))
far = cv2.resize(far, dim)
eye_far = np.int32(eye_far * scale)

base = np.uint8(255 * np.ones(far.shape))


shift = eye_far[0] - eye_near[0]
translate = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
translate_inv = np.linalg.inv(translate)

for i in range(base.shape[0]):
    for j in range(base.shape[1]):
        pnt = translate_inv @ np.array([j, i, 1])
        pnt = np.int32(pnt / pnt[2])
        if np.all(pnt == np.uint32(pnt)):
            try:
                base[i, j, :] = near[pnt[1], pnt[0], :]
            except BaseException:
                pass
near = base
cv2.imwrite('res21-near.jpg', near)
cv2.imwrite('res22-far.jpg', far)

DFT_near = np.array([np.fft.fftshift(np.fft.fft2(near[:, :, 0])),
                     np.fft.fftshift(np.fft.fft2(near[:, :, 1])),
                     np.fft.fftshift(np.fft.fft2(near[:, :, 2]))])

DFT_far = np.array([np.fft.fftshift(np.fft.fft2(far[:, :, 0])),
                    np.fft.fftshift(np.fft.fft2(far[:, :, 1])),
                    np.fft.fftshift(np.fft.fft2(far[:, :, 2]))])


# s : far : low
# r : near : high

s = 15
r = 30
[HPF_near, _] = filter_generate(near, r)
[_, LPF_far] = filter_generate(far, s)

HPF_near_3d = np.array([HPF_near,
                        HPF_near,
                        HPF_near])

LPF_far_3d = np.array([LPF_far,
                       LPF_far,
                       LPF_far])

filtered_near = np.multiply(DFT_near, HPF_near_3d)
filtered_far = np.multiply(DFT_far, LPF_far_3d)

l = 0.6
result_DFT = np.add(l * filtered_near, (1 - l) * filtered_far)


c1 = np.real(np.fft.ifft2(np.fft.ifftshift(result_DFT[0])))
c2 = np.real(np.fft.ifft2(np.fft.ifftshift(result_DFT[1])))
c3 = np.real(np.fft.ifft2(np.fft.ifftshift(result_DFT[2])))
result = cv2.merge([c1, c2, c3])


##### resize and save all results ####

(h, w) = np.shape(result)[0:2]
scale = 4
res_near = cv2.resize(result.copy(), (int(scale * w), int(scale * h)))
scale = .2
res_far = cv2.resize(result.copy(), (int(scale * w), int(scale * h)))

fig = plt.figure()
plt.imshow(HPF_near, 'gray')
plt.axis('off')
fig.savefig('res25-highpass-%d.jpg' % r, dpi=3 * fig.dpi)
plt.close(fig)

fig = plt.figure()
plt.imshow(LPF_far, 'gray')
plt.axis('off')
fig.savefig('res26-lowpass-%d.jpg' % s, dpi=3 * fig.dpi)
plt.close(fig)

my_save(DFT_near, 'res23-dft-near.jpg')
my_save(DFT_far, 'res24-dft-far.jpg')
my_save(filtered_near, 'res27-highpassed.jpg')
my_save(filtered_far, 'res28-lowpassed.jpg')
my_save(result_DFT, 'res29-hybrid.jpg')
cv2.imwrite('res30-hybrid-near.jpg', res_near)
cv2.imwrite('res31-hybrid-far.jpg', res_far)
