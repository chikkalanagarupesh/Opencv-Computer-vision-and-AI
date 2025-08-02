import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def readImage():
    root = os.getcwd()
    filename = 'demoimages//LU1H9206.JPG'
    imgPath = os.path.join(root, filename)
    img = cv2.imread(imgPath)
    width = 640
    height = 480
    img = cv2.resize(img, (width, height))  # Resize image
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def writeImage():
    root = os.getcwd()
    filename = 'demoimages//LU1H9206.JPG'
    imgPath = os.path.join(root, filename)
    img = cv2.imread(imgPath)
    outpath=os.path.join(root,'demoimages//lu1h4070.jpg')
    cv2.imwrite(outpath,img)
if __name__ == "__main__":
    #readImage()
    writeImage()