import cv2
import os
import numpy as np      
import matplotlib.pyplot as plt
def tresholdhing():
    root = os.getcwd()
    imgPath=os.path.join(root, 'demoimages//Prem_picture.JPG')
    img = cv2.imread(imgPath)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([imgGray], [0], None, [256], [0, 256])
    plt.figure()
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

    thresopt=[cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV,
              cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]
    thresNames = ['BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']  

    plt.figure()
    plt.subplot(231)
    plt.imshow(imgGray, cmap='gray')
    plt.title('Original Image')

    for i in range(len(thresopt)):
        ret, imgThres = cv2.threshold(imgGray, 127, 255, thresopt[i])
        plt.subplot(2, 3, i + 2)
        plt.imshow(imgThres, cmap='gray')
        plt.title(thresNames[i])
        plt.axis('off')
    plt.show()
if __name__ == "__main__":
    tresholdhing()


