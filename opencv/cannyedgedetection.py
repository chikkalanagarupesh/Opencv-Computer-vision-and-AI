import os
import cv2
import numpy as np      
import matplotlib.pyplot as plt
def callback(input):
    pass
def cannyEdgeDetection():
    root = os.getcwd()
    imgPath = os.path.join(root, 'demoimages//passportsize.jpeg')
    img = cv2.imread(imgPath)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height,width,_= img.shape
    scale = 1/5
    width = int(width * scale)
    height = int(height * scale)
    img = cv2.resize(img, (width, height),interpolation=cv2.INTER_LINEAR)

    winname = 'Canny Edge Detection'
    cv2.namedWindow(winname)
    cv2.createTrackbar('low', winname, 50, 255, callback)
    cv2.createTrackbar('high', winname, 150, 255, callback)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Get current values of trackbars
        low = cv2.getTrackbarPos('low', winname)
        high = cv2.getTrackbarPos('high', winname)

        # Canny Edge Detection
        edges = cv2.Canny(imgGray, low, high)
        cv2.imshow(winname, edges)  

       
    cv2.destroyAllWindows()
if __name__ == "__main__":
    cannyEdgeDetection()