import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
def convolution2d():
    root = os.getcwd()
    filename = 'demoimages//Prem_picture.JPG'
    imagePath = os.path.join(root,filename)
    img =  cv2.imread(imagePath)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    n=100
    Kernel = np.ones((n,n),np.float32)/(n*n)
    imgFiltered = cv2.filter2D(imgRGB,-1,Kernel)
    plt.figure()

    plt.subplot(121)
    plt.imshow(imgRGB)
    plt.title('Original Image')

    plt.subplot(122)
    plt.imshow(imgFiltered)
    plt.title('Filtered Image')
    
    plt.show()
if __name__ == "__main__":
    convolution2d()