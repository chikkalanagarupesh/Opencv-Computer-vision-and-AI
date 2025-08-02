import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
def callback(input):
    pass
def averageFiltering():
    root=os.getcwd()
    filename = 'demoimages//Prem_picture.JPG'
    imgPath=os.path.join(root, filename)
    img=cv2.imread(imgPath)
    winName = 'avg filter'
    cv2.namedWindow(winName)
    cv2.createTrackbar('n',winName,1,100,callback)
    height,width,_ = img.shape
    scale=1/4
    width=int(width*scale)
    height=int(height*scale)
    img = cv2.resize(img,(width,height))
    while True:
        if cv2.waitKey(1) == ord('q'):
            break
        n= cv2.getTrackbarPos('n',winName)
        imgFilter=cv2.blur(img,(n,n))
        cv2.imshow(winName,imgFilter)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    averageFiltering()
