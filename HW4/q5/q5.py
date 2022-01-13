import cv2
import numpy as np

def capture_points(img_in):
    img=img_in.copy()
    pnts = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pnts.append([x,y])
            if len(pnts)>1:
                cv2.line(img,tuple(pnts[-1]),tuple(pnts[-2]),(0,255,0),2)
            cv2.circle(img, (x,y), 2, (255,0,0), 6)
            cv2.imshow("image", img)

    img=cv2.imread('tasbih.jpg')
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img=cv2.line(img,tuple(pnts[-1]),tuple(pnts[0]),(0,255,0),2)
    return [img,pnts]


def draw_countor(img_in,points):
    img=img_in.copy()
    for i in range(1,len(points)):
        cv2.line(img,points[i-1],points[i],(0,255,0),2)
        cv2.circle(img, points[i-1], 2, (255,0,0), 6)
    cv2.line(img,points[0],points[-1],(0,255,0),2)
    cv2.circle(img, points[-1], 2, (255,0,0), 6)
    return img

def calc_d_bar(points):
    tmp=[]
    for i in range(len(points)-1):
        tmp.append(np.linalg.norm(points[i]-points[i+1]))
    d_bar=np.average(np.array(tmp))
    center=np.average(pnts,axis=0)
    return d_bar , center


def energy_v1_v2(gradient_image,v1, v2,center, d_bar, alpha, beta, gamma):
    l2=np.linalg.norm(np.array(v1)-np.array(v2))
    lc2=np.linalg.norm(np.array(v1)-np.array(center))
    E_ex = -gamma * gradient_image[v1[1], v1[0]]
    E_in = alpha * (l2 - 0.4 * d_bar) ** 2 + beta * lc2
    return E_in + E_ex




img=cv2.imread('tasbih.jpg')
img_blurred = cv2.GaussianBlur(img, (15,15), 20)
K_x = np.array([[1, 0, -1],[1, 0, -1],[1, 0, -1]])
Gx= cv2.filter2D(img_blurred, -1, K_x)
Gy=cv2.filter2D(img_blurred, -1, K_x.T)
img_gradient=np.uint8(Gx**2+Gy**2)
img_gradient=cv2.cvtColor(img_gradient,cv2.COLOR_BGR2GRAY)
mask =(img_gradient < 68)
img_gradient = np.uint8((1-mask)*img_gradient)

img_start,pnts=capture_points(img)

cv2.imwrite('frame1.jpg',img_start)