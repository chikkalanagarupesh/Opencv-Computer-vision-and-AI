import cv2
import numpy as np
import os         
import matplotlib.pyplot as plt
def pureColor():
    zeros=np.zeros((100,100))
    ones=np.ones((100,100))

    bImg=cv2.merge((zeros,zeros,255*ones))
    gImg=cv2.merge((zeros,255*ones,zeros))
    rImg=cv2.merge((255*ones,zeros,zeros))
    plt.figure()

    plt.subplot(231)
    plt.title('Blue Image')
    plt.imshow(bImg)

    plt.subplot(232)     
    plt.title('Green Image')  
    plt.imshow(gImg)

    plt.subplot(233)
    plt.title('Red Image')
    plt.imshow(rImg)

    plt.show()
def bgrChannelGrayscale():
    root = os.getcwd()
    filename = 'demoimages//LU1H9206.JPG'
    imgPath = os.path.join(root, filename)
    img = cv2.imread(imgPath)
    
    b, g, r = cv2.split(img)
    
    plt.figure()
    plt.subplot(131)
    plt.title('Blue Channel')
    plt.imshow(b, cmap='gray')

    plt.subplot(132)
    plt.title('Green Channel')
    plt.imshow(g, cmap='gray')

    plt.subplot(133)
    plt.title('Red Channel')
    plt.imshow(r, cmap='gray')

    plt.show()
def bgrchannelcolor():
    root = os.getcwd()
    filename = 'demoimages//LU1H9206.JPG'
    imgPath = os.path.join(root, filename)
    img = cv2.imread(imgPath)
    b, g, r = cv2.split(img)
    zeros= np.zeros_like(b)
    bimg = cv2.merge((b, zeros, zeros))
    gimg = cv2.merge((zeros, g,zeros))
    rimg = cv2.merge((zeros, zeros, r))

    plt.figure()

    plt.subplot(131)
    plt.title('Blue Channel')
    plt.imshow(bimg)

    plt.subplot(132)
    plt.title('Green Channel')
    plt.imshow(gimg)

    plt.subplot(133)
    plt.title('Red Channel')
    plt.imshow(rimg)

    plt.show()
if __name__ == "__main__":
    bgrchannelcolor()
    #bgrChannelGrayscale()
    #pureColor()
   