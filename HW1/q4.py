import numpy as np
import cv2

img=cv2.imread('Flowers.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
color_low=(140,40,30)
color_high=(170,255,255) 
mask = cv2.inRange(img_hsv, color_low, color_high)/255

img2=img.copy()
img2[:,:,0]=np.multiply(img[:,:,0],mask)
img2[:,:,1]=np.multiply(img[:,:,1],mask)
img2[:,:,2]=np.multiply(img[:,:,2],mask)

img_blur=cv2.blur(img,(21,21)) 
img_blur[:,:,0]=np.multiply(img_blur[:,:,0],1-mask)
img_blur[:,:,1]=np.multiply(img_blur[:,:,1],1-mask)
img_blur[:,:,2]=np.multiply(img_blur[:,:,2],1-mask)
img_blur=np.uint8(img_blur)

#show(img2)

img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
hue=img2_hsv[:,:,0].copy()
hue[np.where(mask)] -=78
img2_hsv[:,:,0]=hue
img2 = cv2.cvtColor(img2_hsv, cv2.COLOR_BGR2HSV)

result=(np.uint8(img_blur+img2))
cv2.imwrite('res06.jpg',result)