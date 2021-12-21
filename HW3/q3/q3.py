import cv2
from function import *
import numpy as np


def hole_fill(img_in,holes,source_in,block_size=150 , overlap=70):
    source=source_in.copy()
    img_hole=img_in.copy()
    pad=np.uint8(np.zeros([block_size,img_hole.shape[1],3]))
    img_hole=cv2.vconcat([pad,img_hole,pad])


    for row in holes.copy():
        row[0][1]+=block_size
        row[1][1]+=block_size

        img_hole[row[0][1]:row[1][1],row[0][0]:row[1][0],:]=0
        output_size=(row[1][0]-overlap)-(row[0][0]-block_size)
        patch_count1 = int(np.ceil((output_size - overlap) / (block_size - overlap)))+1
        pad1=patch_count1*block_size -(patch_count1-1)*overlap - output_size

        output_size=(row[1][1]-overlap)-(row[0][1]-block_size)
        patch_count2 = int(np.ceil((output_size - overlap) / (block_size - overlap)))+1
        pad2=patch_count2*block_size -(patch_count2-1)*overlap - output_size

        blank=img_hole[row[0][1]-block_size:row[1][1]-overlap+pad2,row[0][0]-block_size:row[1][0]-overlap+pad1,:].copy()

        for i in range(2,patch_count2+1):
            for j in range(2,patch_count1+1):
            
                i_v=[(i-1)*block_size-(i-1)*overlap,(i-1)*block_size-(i-2)*overlap]
                i_h=[(j-1)*block_size-(j-1)*overlap,(j-1)*block_size-(j-2)*overlap]
                margin1_up=blank[i_v[0]:i_v[1],i_h[0]:j*block_size-(j-1)*overlap,:].copy()
                margin1_left=blank[i_v[0]:i*block_size-(i-1)*overlap,i_h[0]:i_h[1],:].copy()
                if j==patch_count1:
                    margin_right=blank[i_v[0]:i*block_size-(i-1)*overlap,i_h[0]+overlap:,:].copy()     
                if i==patch_count2:
                    margin_down=blank[i_v[0]+overlap:,i_h[0]:j*block_size-(j-1)*overlap,:].copy()

                patch1=np.zeros([block_size,block_size,3])
                mask=np.zeros([block_size,block_size])

                mask[:overlap,:]=255
                mask[:,:overlap]=255

                patch1[:overlap,:,:]=margin1_up.copy()
                patch1[:,:overlap,:]=margin1_left.copy()

                patch2=random_patch(source,np.uint8(patch1),block_size,overlap,True, np.uint8(mask))

                patch_combined_up=combine_margin(margin1_up,patch2[0:overlap,:,:],'v')
                patch_combined_left=combine_margin(margin1_left,patch2[:,0:overlap,:])
                    
                
                blank[i_v[0]:i*block_size-(i-1)*overlap,i_h[0]:j*block_size-(j-1)*overlap,:]=patch2.copy()
                blank[i_v[0]:i*block_size-(i-1)*overlap,i_h[0]:i_h[1],:]=(patch_combined_left).copy()
                blank[i_v[0]:i_v[1],i_h[0]:j*block_size-(j-1)*overlap,:]=(patch_combined_up).copy()
                
                if j==patch_count1:
                    patch_combined_right=combine_margin(patch2[:,block_size-margin_right.shape[1]:,:],margin_right)
                    blank[i_v[0]:i*block_size-(i-1)*overlap,i_h[0]+overlap:,:]=patch_combined_right

                if i==patch_count2:
                    patch_combined_down=combine_margin(patch2[block_size-margin_down.shape[0]:,:,:],margin_down,'v')
                    blank[i_v[0]+overlap:,i_h[0]:j*block_size-(j-1)*overlap,:]=patch_combined_down
        img_hole[row[0][1]-block_size:row[1][1]-overlap+pad2,row[0][0]-block_size:row[1][0]-overlap+pad1,:]=blank 
    img_hole=img_hole[block_size:img_in.shape[0]+block_size]

    return img_hole




bird=[[[325 , 36],[547 , 165]],
      [[823 , 737], [987, 938]],
      [[1120 ,605],[1250, 730]]]


sea=[[[737 , 688],[957 , 1161]]]



img3=cv2.imread('im03.jpg')
img4=cv2.imread('im04.jpg')


source_bird=img3[bird[0][1][1]:,:bird[1][0][0],:]
source_sea=img4[sea[0][1][1]:,:,:]



img3_fill=hole_fill(img3,bird,source_bird,block_size=150, overlap=70)
img4_fill=hole_fill(img4,sea,source_sea,block_size=150, overlap=70)


cv2.imwrite('res15.jpg',img3_fill)
cv2.imwrite('res16.jpg',img4_fill)
