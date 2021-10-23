
import cv2
import numpy as np
import matplotlib.pyplot as plt


def hist_spec(source,target):
    # cumulative hist for target image
    y, x = np.histogram(target, bins=np.arange(256), density=True)
    H2 = np.float64(np.cumsum(y) / np.sum(y))
    
    # cumulative hist for source image
    y, x = np.histogram(source, bins=np.arange(256), density=True)
    H1 = np.float64(np.cumsum(y) / np.sum(y))
    
    # calc H2_inv(H1(r))
    tmp = np.interp(source.flatten(), x[:-1], H1)
    source_ = np.interp(tmp.flatten(), H2, x[:-1])
    source_ = source_.reshape(np.shape(source)[0:2])
    return source_


pink = cv2.imread('Pink.JPG')
dark = cv2.imread('Dark.JPG')


dark2=dark.copy()
dark2[:,:,0]=hist_spec(dark[:,:,0],pink[:,:,0])
dark2[:,:,1]=hist_spec(dark[:,:,1],pink[:,:,1])
dark2[:,:,2]=hist_spec(dark[:,:,2],pink[:,:,2])
cv2.imwrite('res11.jpg', dark2)



y1, x = np.histogram(dark2[:,:,0], bins=np.arange(256)); y1 = y1 / np.sum(y1)
y2, x = np.histogram(dark2[:,:,1], bins=np.arange(256)); y2 = y2 / np.sum(y2)
y3, x = np.histogram(dark2[:,:,2], bins=np.arange(256)); y3 = y3 / np.sum(y3)

fig = plt.figure(figsize=(36, 24))
plt.subplot(234); plt.plot(x[:-1], y1); plt.grid()
plt.ylabel('normalized value', fontsize='x-large')
plt.xlabel('Pixel value', fontsize='x-large')
plt.title('hsitogram for Dark.jpg Blue channel after hist specificatin',fontsize='xx-large')
plt.subplot(235); plt.plot(x[:-1], y2); plt.grid()
plt.ylabel('normalized value', fontsize='x-large')
plt.xlabel('Pixel value', fontsize='x-large')
plt.title('hsitogram for Dark.jpg Green channel after hist specificatin',fontsize='xx-large')
plt.subplot(236); plt.plot(x[:-1], y3); plt.grid()
plt.ylabel('normalized value', fontsize='x-large')
plt.xlabel('Pixel value', fontsize='x-large')
plt.title('hsitogram for Dark.jpg Red channel after hist specificatin',fontsize='xx-large')



y1, x = np.histogram(dark[:,:,0], bins=np.arange(256)); y1 = y1 / np.sum(y1)
y2, x = np.histogram(dark[:,:,1], bins=np.arange(256)); y2 = y2 / np.sum(y2)
y3, x = np.histogram(dark[:,:,2], bins=np.arange(256)); y3 = y3 / np.sum(y3)

plt.subplot(231); plt.plot(x[:-1], y1); plt.grid()
plt.ylabel('normalized value', fontsize='x-large')
plt.xlabel('Pixel value', fontsize='x-large')
plt.title('hsitogram for Dark.jpg Blue channel defore hist specificatin',fontsize='xx-large')
plt.subplot(232); plt.plot(x[:-1], y2); plt.grid()
plt.ylabel('normalized value', fontsize='x-large')
plt.xlabel('Pixel value', fontsize='x-large')
plt.title('hsitogram for Dark.jpg Green channel defore hist specificatin',fontsize='xx-large')
plt.subplot(233); plt.plot(x[:-1], y3); plt.grid()
plt.ylabel('normalized value', fontsize='x-large')
plt.xlabel('Pixel value', fontsize='x-large')
plt.title('hsitogram for Dark.jpg Red channel defore hist specificatin',fontsize='xx-large')

fig.savefig('res10.jpg', dpi=3 * fig.dpi)
plt.close(fig)
