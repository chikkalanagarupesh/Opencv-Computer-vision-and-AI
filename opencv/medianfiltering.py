import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
def medianFiltering():
    root = os.getcwd()
    filename = 'demoimages//Prem_picture.JPG'
    imgPath = os.path.join(root, filename)      
    img = cv2.imread(imgPath)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    noisyImg= imgRGB.copy()
    noiseProb=0.05
    noise = np.random.rand(noisyImg.shape[0],noisyImg.shape[1])
    noisyImg[noise < noiseProb]=0
    noisyImg[noise>1-noiseProb]=255
    imgFiltered = cv2.medianBlur(noisyImg,5)
    plt.figure()
    plt.subplot(121)
    plt.imshow(noisyImg)

    plt.subplot(122)
    plt.imshow(imgFiltered)
    plt.show()
if __name__ == "__main__":
    medianFiltering()