import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
def grayHistogram():
    root = os.getcwd()
    filename = 'demoimages//LU1H9206.JPG'
    imgPath = os.path.join(root, filename)
    img = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)

    plt.figure()
    plt.imshow(img, cmap='gray')
    
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    plt.figure()
    plt.plot(hist)
    plt.title('Grayscale Histogram')
    plt.xlabel('bins')
    plt.ylabel('# of pixels')
    plt.show()
def colorHistogram():
    root = os.getcwd()
    filename = 'demoimages//LU1H9206.JPG'
    imgPath = os.path.join(root, filename)
    img = cv2.imread(imgPath)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(imgRGB)
    
    colors = ('b', 'g', 'r')
    plt.figure()
    for i in range (len(colors)):
        hist = cv2.calcHist([imgRGB ], [i], None, [256], [0, 256])
        plt.plot(hist, colors[i])
    
    plt.title('Color Histogram')
    plt.xlabel('bins')
    plt.ylabel('# of pixels')
    plt.show()
def histogramRegion():
    root = os.getcwd()
    filename = 'demoimages//LU1H9206.JPG'
    imgPath = os.path.join(root, filename)
    img = cv2.imread(imgPath)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB = imgRGB[286:345, 322:389, :]
    plt.figure()
    plt.imshow(imgRGB)
    
    colors = ('b', 'g', 'r')
    plt.figure()
    for i in range (len(colors)):
        hist = cv2.calcHist([imgRGB ], [i], None, [256], [0, 256])
        plt.plot(hist, colors[i])
    
    plt.title('Color Histogram of Region')
    plt.xlabel('bins')
    plt.ylabel('# of pixels')
    plt.show()
if __name__ == "__main__":
    #grayHistogram()
    #colorHistogram()
    histogramRegion()