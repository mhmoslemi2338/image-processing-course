import numpy as np
import matplotlib.pyplot as plt

def kmean(k,arr):
    epsilon=0.0001
    index=np.random.randint(0,len(arr),k)
    centers=arr[index]
    while True:
        clusters=np.zeros((k,0)).tolist()
        for i,row in enumerate(arr):
            dist=np.sum(np.power(np.array(centers)-row,2),axis=1)
            clusters[np.argmin(dist)].append(row)
        centers_new=np.zeros((k,2))
        for i in range(k):
            centers_new[i]=np.mean(clusters[i],axis=0)
        if np.sum(np.power(centers_new/centers-1,2)) < epsilon : 
            break
        else:
            centers=centers_new.copy()
    return centers_new , clusters


#### import points #####
points=[]
with open('Points.txt') as f:
    lines = f.readlines()
for row in lines[1:]:
    [p1,p2]=row.rstrip().split(' ')
    points.append([float(p1),float(p2)])
points=np.array(points)


### k-mean algorithm ####
centers1,clusters1=kmean(2,points)
centers2,clusters2=kmean(2,points)


##### plot and save result #######
fig=plt.figure(figsize=[12,12])
plt.scatter(points[:,0],points[:,1],s=15)
plt.axis('off')
fig.savefig('res01.jpg', dpi=4 * fig.dpi)
plt.close(fig)

fig=plt.figure(figsize=[12,12])
plt.scatter(np.array(clusters1[0])[:,0],np.array(clusters1[0])[:,1],c='red',s=15)
plt.scatter(np.array(clusters1[1])[:,0],np.array(clusters1[1])[:,1],c='blue',s=15)
c1=plt.scatter(centers1[0,0],centers1[0,1],marker='^',c='red',s=100,label='c1')
c2=plt.scatter(centers1[1,0],centers1[1,1],marker='^',c='blue',s=100,label='c2')
plt.axis('off'); plt.legend(loc='best',handles=[c1,c2],labels=['Cluster center1','Cluster center2'])
fig.savefig('res02.jpg', dpi=4 * fig.dpi)
plt.close(fig)

fig=plt.figure(figsize=[12,12])
plt.scatter(np.array(clusters2[0])[:,0],np.array(clusters2[0])[:,1],c='red',s=15)
plt.scatter(np.array(clusters2[1])[:,0],np.array(clusters2[1])[:,1],c='blue',s=15)
c1=plt.scatter(centers2[0,0],centers2[0,1],marker='^',c='red',s=100,label='c1')
c2=plt.scatter(centers2[1,0],centers2[1,1],marker='^',c='blue',s=100,label='c2')
plt.axis('off'); plt.legend(loc='best',handles=[c1,c2],labels=['Cluster center1','Cluster center2'])
fig.savefig('res03.jpg', dpi=4 * fig.dpi)
plt.close(fig)

####**********************************************####
#### change feature space from cartesian to polar ####
####**********************************************####
X=points[:,0]
Y=points[:,1]
rho=np.sqrt(X**2+Y**2)
theta=np.arctan(Y/X)
points2=np.float32(np.vstack((rho, theta)).T)

### k-mean algorithm ####
centers1,clusters1=kmean(2,points2)
centers2,clusters2=kmean(2,points2)

##### plot and save result #######
fig=plt.figure(figsize=[36,12])
plt.subplot(131)
plt.scatter(points2[:,0],points2[:,1],s=15)
plt.subplot(132)
plt.scatter(np.array(clusters1[0])[:,0],np.array(clusters1[0])[:,1],c='red',s=15)
plt.scatter(np.array(clusters1[1])[:,0],np.array(clusters1[1])[:,1],c='blue',s=15)
c1=plt.scatter(centers1[0,0],centers1[0,1],marker='^',c='red',s=100,label='c1')
c2=plt.scatter(centers1[1,0],centers1[1,1],marker='^',c='blue',s=100,label='c2')
plt.legend(loc='best',handles=[c1,c2],labels=['Cluster center1','Cluster center2'])
plt.subplot(133)
plt.scatter(np.array(clusters2[0])[:,0],np.array(clusters2[0])[:,1],c='red',s=15)
plt.scatter(np.array(clusters2[1])[:,0],np.array(clusters2[1])[:,1],c='blue',s=15)
c1=plt.scatter(centers2[0,0],centers2[0,1],marker='^',c='red',s=100,label='c1')
c2=plt.scatter(centers2[1,0],centers2[1,1],marker='^',c='blue',s=100,label='c2')
plt.legend(loc='best',handles=[c1,c2],labels=['Cluster center1','Cluster center2'])
fig.savefig('res04.jpg', dpi=4 * fig.dpi)
plt.close(fig)