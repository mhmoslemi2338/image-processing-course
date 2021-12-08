import cv2
import numpy as np
from scipy.signal import argrelextrema


import os
import pickle
def readvar(name):
    name='variables/'+name+'.pckl'  
    f = open(name, 'rb')
    myvar = pickle.load(f)
    f.close()
    return myvar


def hough_space(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0) 
    gray=cv2.resize(gray , (np.min(gray.shape),np.min(gray.shape)) )


    edge = cv2.Canny(gray, threshold1=70, threshold2=110)
    edge_h, edge_w = edge.shape

    d=int(np.sqrt(edge_h**2+edge_w**2))


    thetas = np.linspace(-np.pi, np.pi, 360)
    rhos = np.linspace(-d, d, 2*d)

    accumulator=np.zeros([len(rhos),len(thetas)])

    for y in range(edge_h):
        for x in range(edge_w):
            if edge[y,x]!=255: continue  

            r = x * np.cos(thetas) + y * np.sin(thetas)      
            for i,row in enumerate(r):
                r_index=np.argmin(np.abs(rhos-row))  
                accumulator[r_index][i]+=1
    return accumulator
        




def find_lines(img,accumulator,thresh=150):
    h=np.min(img.shape[0:2])
    d=int(np.sqrt(h**2+h**2))
    thetas = np.linspace(-np.pi, np.pi, 360)
    rhos = np.linspace(-d, d, 2*d)
    
    res=argrelextrema(accumulator.reshape(-1),np.greater)
    (X,Y)=np.unravel_index(res[0], accumulator.shape)
    lines=[]
    for i,x in enumerate(X):
        y=Y[i]
        if accumulator[x][y]<thresh: continue
        rho=rhos[x]
        theta=thetas[y]

        if np.abs(np.abs(theta)-np.pi/2) < np.deg2rad(0.2):
            lines.append([rho,np.inf,theta])
            continue
        if np.abs(np.abs(theta)-np.pi) < np.deg2rad(0.2):
            lines.append([rho,-np.inf,theta])
            continue
        
        b=rho/np.sin(theta)
        m=-np.cos(theta)/np.sin(theta) 
        lines.append([m,b,theta])
        
    return lines





def draw_line(im,lines): 
    img=cv2.resize(im.copy() , (np.min(im.shape[0:2]),np.min(im.shape[0:2])) )
    for row in lines:
        [m,b,_]=row
        if b==np.inf :  
            cv2.line(img,(-5000,round(m)) ,(5000,round(m)) , (0, 0,255 ), thickness=1) 
            continue
        if b==-np.inf:
            cv2.line(img,(-round(m),-5000) ,(-round(m),5000) , (0, 0, 255), thickness=1)
            continue

        x1=-4000
        y1=int(m*x1+b)
        x2=4000
        y2=int(m*x2+b)
        cv2.line(img,(x1,y1) ,(x2,y2) , (0, 0, 255), thickness=1)          
    return img




  

for i in range(1,21):
# for i in range(13,14):
    try:
        name='hough/im%0.2d.jpg'%(i)
        im=cv2.imread(name)
        accumulator=readvar('accumulator'+str(i))
        lines=find_lines(im,accumulator,thresh=120)
        im=draw_line(im,lines)
        cv2.imwrite('res/res%d.jpg'%i,im)
        
    except:
        print(i)



 



# im1=cv2.imread('im01.jpg')
# accumulator=hough_space(im1)
# lines=find_lines(im1,accumulator)
# im1=draw_line(im1,lines) 
# cv2.imwrite('tmp1.jpg',im1)

  
# im1=cv2.imread('im02.jpg')
# accumulator=hough_space(im1)
# lines=find_lines(im1,accumulator)
# im1=draw_line(im1,lines) 
# cv2.imwrite('tmp2.jpg',im1)

