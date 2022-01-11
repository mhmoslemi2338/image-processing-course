import numpy as np
from skimage import morphology
import cv2
import copy


NAMES=['res06.jpg','res07.jpg','res08.jpg','res09.jpg']
K=[64,256,1024,2048]
alpha=0.25

### prepare image and Gradient of image
img=cv2.imread('slic.jpg')
img_lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
K_x = np.array([[1, 0, -1],[1, 0, -1],[1, 0, -1]])
Gx= cv2.filter2D(img_gray, -1, K_x)
Gy=cv2.filter2D(img_gray, -1, K_x.T)
img_gradient=(Gx**2+Gy**2)
H,W = img_gray.shape


for rnd,k in enumerate(K):
    ### find initial centroids
    S=round(np.sqrt(H*W/k))
    centroids=[]
    X=np.arange(S//2,H,S)
    Y=np.arange(S//2,W,S)
    centroids=np.array(np.meshgrid(X,Y)).T.reshape(-1,2)

    ### move initial centroids to lowest local gradient
    for i, row in enumerate([centroids[12]]):
        patch=img_gradient[row[0]-2:row[0]+3,row[1]-2:row[1]+3]
        if img_gradient[row[0],row[1]] != np.min(patch) :
            x,y=np.where(patch==np.min(patch))
            tmp=[]
            for ii in range(len(x)):
                idx=(x[ii],y[ii])
                local=patch[max(x[ii]-1,0):x[ii]+2,max(y[ii]-1,0):y[ii]+2]
                tmp.append([np.sum(local)/len(local.reshape(-1)),ii])
            tmp.sort(key=lambda x:x[0])
            idx=tmp[0][1]
            dx,dy=x[idx]-2,y[idx]-2
            centroids[i]=(row[0]+dx , row[1]+dy)
    centroids=np.int64(centroids)

    ### make feature vector #####
    feature = (np.zeros((H,W,5)))
    feature[:,:,0:3]=copy.deepcopy(img_lab)
    feature[:,:,3]=(np.array([np.arange(0,H)]).T)
    feature[:,:,4]=(np.array([np.arange(0,W)]))

    dis_mat=np.full(img_gray.shape,10000000)
    label = (-1)*np.ones(img_gray.shape)
    while(True):
        flag=True
        for index,center in enumerate(centroids,start=1):
            #### calculate distance between pixels and center of cluster
            x_range=(max(center[0]-S,0),min(center[0]+S,H)+1)
            y_range=(max(center[1]-S,0),min(center[1]+S,W)+1)

            pnts=feature[x_range[0]:x_range[1],y_range[0]:y_range[1],:]
            lab_center=(img_lab[center[0], center[1]])
            xy_center=np.array((center[0],center[1]))

            D_lab=np.linalg.norm(pnts[:,:,0:3]-lab_center,axis=2)
            D_xy=np.linalg.norm(pnts[:,:,3:]-xy_center,axis=2)
            D = D_lab + alpha * D_xy

            ### assign label to pixels in windows
            mask=(D<dis_mat[x_range[0]:x_range[1],y_range[0]:y_range[1]])
            dist_old=(dis_mat[x_range[0]:x_range[1],y_range[0]:y_range[1]])*(1-mask)
            dis_mat[x_range[0]:x_range[1],y_range[0]:y_range[1]]= D*mask
            dis_mat[x_range[0]:x_range[1],y_range[0]:y_range[1]]+=dist_old

            label_old=label[x_range[0]:x_range[1],y_range[0]:y_range[1]]*(1-mask)
            label[x_range[0]:x_range[1],y_range[0]:y_range[1]]=mask*index
            label[x_range[0]:x_range[1],y_range[0]:y_range[1]]+=label_old

            #### update cluster center
            cluster_mask=(label[x_range[0]:x_range[1],y_range[0]:y_range[1]]==index)
            delta=np.average(np.argwhere(cluster_mask),axis=0)
            new_center=np.int32(np.array((x_range[0],y_range[0]))+delta)

            if np.linalg.norm(100*new_center/center-100) > 0.001 :
                centroids[index-1]=new_center.copy()
                flag=False
        if flag:
            break


    label=morphology.closing(label,np.ones((30,30)))
    boundaries = np.uint8(cv2.Laplacian(label, -1, ksize=3)) > 1
    boundaries=cv2.merge([1-boundaries,1-boundaries,1-boundaries])
    res=img*boundaries
    cv2.imwrite(NAMES[rnd],res)
