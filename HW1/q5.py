import cv2
import numpy as np
import timeit

#### method one
img=cv2.imread('Pink.jpg')
kernel=1/9*np.ones((3,3))
start = timeit.default_timer()

img2 = cv2.filter2D(src=img, kernel=kernel, ddepth=-1)

stop = timeit.default_timer()
print('Run-Time for method one : %0.4f sec'%(stop - start))
cv2.imwrite('res07.jpg',img2)

#### method two
img=cv2.imread('Pink.jpg')
img2=img.copy()
(h,w,_)=np.shape(img2)

for i in range(h-2):
    for j in range(w-2):
        img2[i+1,j+1,0]=int(np.sum(img[i:i+3,j:j+3,0])/9)
        img2[i+1,j+1,1]=int(np.sum(img[i:i+3,j:j+3,1])/9)
        img2[i+1,j+1,2]=int(np.sum(img[i:i+3,j:j+3,2])/9)
        
res=np.uint8(img2)
stop = timeit.default_timer()
print('Run-Time for method two : %0.3f sec' %(stop - start))
cv2.imwrite('res08.jpg',img2)

