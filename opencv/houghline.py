import os
import cv2
import numpy as np  
import matplotlib.pyplot as plt
def houghLineTransform():
    root = os.getcwd()      
    imgPath = os.path.join(root, 'demoimages//Prem_picture.JPG')
    img = cv2.imread(imgPath,   cv2.IMREAD_GRAYSCALE)
    imgBlur= cv2.GaussianBlur(img, (21, 21), 3)
    Cannyedges = cv2.Canny(imgBlur, 50, 180)
    plt.figure()
    plt.subplot(141)
    plt.imshow(img)
    plt.subplot(142)
    plt.imshow(imgBlur)
    plt.subplot(143)
    plt.imshow(Cannyedges)

    distResol=1
    angleResol=np.pi/180
    threshold=100
    lines= cv2.HoughLines(Cannyedges, distResol, angleResol, threshold)
    k=3000
    for curline in lines:
        rho, theta = curline[0]
    dhat=np.array([[np.cos(theta)],[np.sin(theta)]])
    d=rho*dhat
    lhat=np.array([[-np.sin(theta)],[np.cos(theta)]])
    p1=d-k*lhat
    p2=d+k*lhat 
    p1=p1.astype(int)
    p2=p2.astype(int)
    cv2.line(img,(p1[0][0], p1[1][0]), (p2[0][0], p2[1][0]), (255,255,255),10)
    plt.subplot(144)
    plt.imshow(img)


    plt.show()
if __name__ == "__main__":
    houghLineTransform()