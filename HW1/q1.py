
import cv2
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread('Enhance1.JPG')

a=0.01

img_b=np.array(img[:,:,0],dtype=np.float64)
img_g=np.array(img[:,:,1],dtype=np.float64)
img_r=np.array(img[:,:,2],dtype=np.float64)

img[:,:,0]=np.uint8(np.log10(a*img_b+1)*(255/np.log10(1+255*a)))
img[:,:,1]=np.uint8(np.log10(a*img_g+1)*(255/np.log10(1+255*a)))
img[:,:,2]=np.uint8(np.log10(a*img_r+1)*(255/np.log10(1+255*a)))



img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
value=img[:,:,2].flatten()  ### get brigtness channel


#### make hist for brigtness channel
hist=np.zeros(256)
for i in value:
    hist[i]+=1

#### make cummulative func. for hist.
H1=np.zeros(256)
H1[0]=hist[0]
for i in range(1,256):
    H1[i]=H1[i-1]+hist[i]
    
#### normalize hist. & cumulative func.
hist=np.float64(hist)/H1[-1]
H1=np.float64(H1)/H1[-1]


for i,row in enumerate(value):
    value[i]=np.uint8(255*H1[row])


img[:,:,2]=value.reshape(np.shape(img)[0:2])
img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

cv2.imwrite('res01.jpg',img)

