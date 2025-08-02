import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
def hsvColorSpace():
    root=os.getcwd()
    filename = 'demoimages//LU1H9206.JPG'
    imgPath = os.path.join(root, filename)
    img = cv2.imread(imgPath)  
    hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerBound = np.array([0, 50, 50])
    upperBound = np.array([10, 255, 255])   
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    plt.figure()
    plt.imshow(img)
    plt.title('Rupesh')
    plt.show()
    cv2.imshow('mask', mask)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
if __name__ == "__main__":
    hsvColorSpace()