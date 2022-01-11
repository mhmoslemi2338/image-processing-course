


import cv2
import numpy as np
import copy


# filter to reduce noise
img=cv2.imread('park.jpg')
img = cv2.medianBlur(img, 5)
img = cv2.GaussianBlur(img, (5,5),3)
scale_percent = 0.25
w = int(img.shape[1] * scale_percent )
h = int(img.shape[0] * scale_percent )
img = cv2.resize(img, (w, h))



feature = np.float32(img.reshape((-1, 3)))
tmp_h=np.float32(255*np.array(w*(np.arange(0,h).tolist())).reshape(1,-1)/h)
tmp_w=np.float32(255*np.array(h*(np.arange(0,w).tolist())).reshape(1,-1)/w)
feature=np.float32(np.concatenate((feature.T,tmp_h,tmp_w))).T



clusters=[]
thresh=70

while len(feature)>0:
    idx=np.random.randint(0,len(feature))
    mean=feature[idx]
    while True:
        dist=np.linalg.norm(feature-mean,axis=1)
        region_idx=np.where(dist<thresh)[0]
        mean_new=np.mean(feature[region_idx],axis=0)

        if np.linalg.norm(mean_new-mean) < 0.01 :
            if len(clusters)>0:
                for c,row in enumerate(clusters):
                    if np.linalg.norm(np.mean(row,axis=0)-mean_new) < 0.5*thresh :
                        clusters[c]=np.concatenate(( clusters[c],feature[region_idx]))
                        tmp=copy.deepcopy(feature.T)
                        tmp = np.delete(tmp, region_idx,axis=1)
                        feature=copy.deepcopy(tmp.T)
                        break

            clusters.append(copy.deepcopy(feature[region_idx]))
            tmp=copy.deepcopy(feature.T)
            tmp = np.delete(tmp, region_idx,axis=1)
            feature=copy.deepcopy(tmp.T)
            break
        else:
            mean=mean_new.copy()


clusters_mean=[]
for row in clusters:
    clusters_mean.append(np.mean(row,axis=0))


(H,W)=img.shape[0:2]
segments=[[] for i in range(len(clusters_mean))]
for h in range(H):
    for w in range(W):
        feature=np.float32(np.concatenate((img[h,w,:],[255*h/H],[255*w/W])))
        idx=np.argmin(np.linalg.norm(clusters_mean-feature,axis=1))
        segments[idx].append((h,w))


res=0*copy.deepcopy(img)
for i,seg in enumerate(segments):
    for (h,w) in seg:
        res[h,w,:]=np.uint8(clusters_mean[i][0:3])

cv2.imwrite('res05.jpg',res)