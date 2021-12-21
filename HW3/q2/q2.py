import cv2
import numpy as np
from function import *

def main(name,output_size=2500,block_size=150,overlap=40):
    img=cv2.imread(name)
    (h,w,_)=img.shape
    patch_count = int(np.ceil((output_size - overlap) / (block_size - overlap)))
    pad=patch_count*block_size -(patch_count-1)*overlap - output_size
    blank=np.uint8(np.zeros([output_size+pad,output_size+pad,3]))

    x=round(np.random.uniform(high=h-block_size-2))
    y=round(np.random.uniform(high=w-block_size-2))
    patch0=img[x:x+block_size,y:y+block_size,:]
    blank[:block_size,0:block_size,:]=patch0.copy()

    for i in range(1,500):
        try:
            ##### fill first row
            margin1=blank[:block_size,i*block_size-i*overlap:i*block_size-(i-1)*overlap,:].copy()
            patch2=random_patch(img,margin1,block_size,overlap)
            margin2=patch2[:,0:overlap,:].copy()
            patch_combined=combine_margin(margin1,margin2)
            blank[:block_size,i*block_size-i*overlap:(i+1)*block_size-i*overlap,:]=patch2.copy()
            blank[:block_size,i*block_size-i*overlap:i*block_size-(i-1)*overlap,:]=patch_combined.copy()
            ##### fill first column
            margin1=blank[i*block_size-i*overlap:i*block_size-(i-1)*overlap,:block_size,:].copy()
            patch2=random_patch(img,margin1,block_size,overlap)
            margin2=patch2[0:overlap,:,:].copy()
            patch_combined=combine_margin(margin1,margin2,'v')
            blank[i*block_size-i*overlap:(i+1)*block_size-i*overlap,:block_size,:]=patch2.copy()
            blank[i*block_size-i*overlap:i*block_size-(i-1)*overlap,:block_size,:]=patch_combined.copy()
        except: break


    for i in range(2,500):
        for j in range(2,500):
            try:
                i_v=[(i-1)*block_size-(i-1)*overlap,(i-1)*block_size-(i-2)*overlap]
                i_h=[(j-1)*block_size-(j-1)*overlap,(j-1)*block_size-(j-2)*overlap]
                margin1_up=blank[i_v[0]:i_v[1],i_h[0]:j*block_size-(j-1)*overlap,:].copy()
                margin1_left=blank[i_v[0]:i*block_size-(i-1)*overlap,i_h[0]:i_h[1],:].copy()

                patch1=np.zeros([block_size,block_size,3])
                mask=np.zeros([block_size,block_size])
                mask[:overlap,:]=255
                mask[:,:overlap]=255

                patch1[:overlap,:,:]=margin1_up.copy()
                patch1[:,:overlap,:]=margin1_left.copy()
                

                patch2=random_patch(img,np.uint8(patch1),block_size,overlap,True, np.uint8(mask))
                patch_combined_up=combine_margin(margin1_up,patch2[0:overlap,:,:],'v')
                patch_combined_left=combine_margin(margin1_left,patch2[:,0:overlap,:])

                blank[i_v[0]:i*block_size-(i-1)*overlap,i_h[0]:j*block_size-(j-1)*overlap,:]=patch2
                blank[i_v[0]:i*block_size-(i-1)*overlap,i_h[0]:i_h[1],:]=(patch_combined_left)
                blank[i_v[0]:i_v[1],i_h[0]:j*block_size-(j-1)*overlap,:]=(patch_combined_up)
            except: break
    blank=blank[0:output_size,0:output_size,:]
    original=np.uint8(np.ones([output_size,output_size,3])*255)
    original[output_size//2-h//2:output_size//2-h//2+h,output_size//2-w//2:output_size//2-w//2+w,:]=img


    return [blank , original]



[synthes2, original2] = main('texture02.png', output_size=2500, block_size=150, overlap=70)
res2 = cv2.hconcat([original2, synthes2])
cv2.imwrite('res11.jpg', res2)

[synthes6, original6] = main('texture05.jpg', output_size=2500, block_size=150, overlap=70)
res6 = cv2.hconcat([original6, synthes6])
cv2.imwrite('res12.jpg', res6)

[synthes3, original3] = main('Textture_sample_3.jpg', output_size=2500, block_size=150, overlap=70)
res3 = cv2.hconcat([original3, synthes3])
cv2.imwrite('res13.jpg', res3)

[synthes4, original4] = main('Textture_sample_4.jpg', output_size=2500, block_size=150, overlap=70)
res4 = cv2.hconcat([original4, synthes4])
cv2.imwrite('res14.jpg', res4)





