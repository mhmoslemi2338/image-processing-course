import numpy as np
import cv2

  
img = cv2.imread('Flowers.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


l1 = (140,  15, 100)
h1 = (165, 200, 255)
mask1 =cv2.inRange(img_hsv, l1, h1)

l2=(145,  9, 0)
h2=(165,  46, 170)
mask2 =cv2.inRange(img_hsv, l2, h2)


mask=np.bitwise_and(np.bitwise_not(mask2),mask1)
mask=cv2.medianBlur(mask,5)/255


# apply mask to image and select pink areas
img2 = img.copy()
img2[:, :, 0] = np.multiply(img[:, :, 0], mask)
img2[:, :, 1] = np.multiply(img[:, :, 1], mask)
img2[:, :, 2] = np.multiply(img[:, :, 2], mask)


# first blur whole image with avg filter with size 21
# then multiply (1-mask) to image so we select all 
# colors except pink areas but we blurred them already
# pink areas are set to zero or pure black
img_blur = cv2.blur(img, (21, 21))
img_blur=np.multiply(img_blur,cv2.merge((1-mask, 1-mask, 1-mask)))



# make another HSV clone from img2 witch is only pink areas of original image
#  and subtract 78 from channel hue in pink areas to convert pink to yellow
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
hue = img2_hsv[:, :, 0].copy()
hue[np.where(mask)] -= 80
img2_hsv[:, :, 0] = hue
img2 = cv2.cvtColor(img2_hsv, cv2.COLOR_BGR2HSV)
# img2 is pink areas witvh is converted to yellow


result = (np.uint8(img_blur + img2))
cv2.imwrite('res06.jpg', result)

