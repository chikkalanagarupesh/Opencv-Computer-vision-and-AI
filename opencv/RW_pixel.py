import cv2
import os
import matplotlib.pyplot as plt
def readAndwritesinglepixel():
    root=os.getcwd()
    filename='demoimages//LU1H9206.jpg'
    imgPath=os.path.join(root,filename)
    img=cv2.imread(imgPath)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(imgRGB)
    plt.title('RUPESH')
    plt.show()
    eyePixel=img[3000,4000]
    imgRGB[3000,4000]=[0,0,255]
    plt.imshow(imgRGB)
    plt.title('RUPESH') 
    plt.show()
def readAndwritePixelRegion():
    root=os.getcwd()
    filename='demoimages//LU1H9206.jpg'
    imgPath=os.path.join(root,filename)
    img=cv2.imread(imgPath)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(imgRGB)
    plt.title('RUPESH')
    plt.show()
    eyeRegion=img[3000:4000,4000:5000]
    imgRGB[3000:4000,4000:5000]=[0,255,0]
    plt.imshow(imgRGB)
    plt.title('RUPESH') 
    plt.show()
if __name__ == "__main__":
    #readAndwritesinglepixel()
    readAndwritePixelRegion()