import numpy as np
import cv2
from skimage.segmentation import felzenszwalb

### import and resize image to 1/4 of size
img=cv2.imread('birds.jpg')
scale_percent = .25
w = int(img.shape[1] * scale_percent )
h = int(img.shape[0] * scale_percent )
img = cv2.resize(img, (w, h))

### over segmentat image with felzenszwalb method
labels_fz = felzenszwalb((img/255), scale=200, sigma=1, min_size=125)

### calculate gaussian and laplacian blured of image
filter_size=11
gaussian=cv2.GaussianBlur(img,(filter_size,filter_size),sigmaX=10)
LOG = cv2.Laplacian(gaussian, -1, ksize=5)


##### make RFS filter-bank but only with one scale
RFS={}
thetas = [0, 30, 60, 90, 120, 150 ,180]
(sigma_x , sigma_y) = (1, 3)

x=np.linspace(- filter_size // 2, filter_size // 2, num=filter_size)
G1d_x = cv2.getGaussianKernel(filter_size, sigma_x)/sigma_x
G1d_y = cv2.getGaussianKernel(filter_size, sigma_y)/sigma_y
G1d = np.outer(x*G1d_x.T,x*G1d_y.T)

G2d_x=((x**2-(sigma_x**2))*G1d_x)
G2d_y=((x**2-(sigma_y**2))*G1d_y)
G2d=(G2d_x*G2d_y.T)

for idx1,theta in enumerate(thetas):
    rot_mat = cv2.getRotationMatrix2D((filter_size // 2, filter_size // 2), angle=theta,scale=1)
    G1d_rot = cv2.warpAffine(G1d.copy(), rot_mat, G1d.shape, flags=cv2.WARP_INVERSE_MAP)
    G2d_rot = cv2.warpAffine(G2d.copy(), rot_mat, G2d.shape, flags=cv2.WARP_INVERSE_MAP)
    G2d_rot=G2d_rot/np.sum(G2d_rot)
    RFS[(0,idx1)]=G1d_rot.copy()
    RFS[(1,idx1)]=G2d_rot.copy()



#### calculate maximum Response 4 (MR4) #####
tmp0 = np.float64(np.zeros(list(img.shape)+[len(thetas)]))
tmp1 = np.float64(np.zeros(list(img.shape)+[len(thetas)]))
for i in range(len(thetas)):
    tmp0[:,:,:,i]=(cv2.filter2D(np.float64(img), -1, np.float64(RFS[(0,i)])))
    tmp1[:,:,:,i]=(cv2.filter2D(np.float64(img), -1, np.float64(RFS[(1,i)])))

response = [(255*(np.max(tmp0,axis=3))/np.max((np.max(tmp0,axis=3)))),
            (255*(np.max(tmp1,axis=3))/np.max((np.max(tmp1,axis=3)))),
            (gaussian),(LOG)]


##### find intersection of accepted superpixels
range_accept = [(70, 89), (116, 150) , (95, 125) , (30, 95)] 
accepted=[[],[],[],[]]
for label in np.unique(labels_fz.ravel()):
    mask = (labels_fz == label)
    for i in range(4):
        if  range_accept[i][0] < np.average(response[i][mask]) < range_accept[i][1]:
            accepted[i].append(label)
                   
birds = set.intersection(set(accepted[0]),set(accepted[1]),set(accepted[2]),set(accepted[3]))


#### draw final boundaries of selected regions(birds) on image
Result = (0*img).copy()
for label in birds:
    Result[(labels_fz == label)] = img[(labels_fz == label)]
is_labeled = (Result != 0)
boundaries=cv2.Laplacian(np.uint8(is_labeled), -1 ) > 0
boundaries = cv2.morphologyEx(np.uint8(255*boundaries), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
boundaries = cv2.dilate(boundaries,np.ones((3,3),np.uint8))
res=np.uint8(img*(1-boundaries/255))
cv2.imwrite('res10.jpg',res)