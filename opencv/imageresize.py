import cv2 
import os
import matplotlib.pyplot as plt
def imageResize():
    root = os.getcwd()
    filename = 'demoimages//LU1H9206.JPG'
    imgPath = os.path.join(root, filename)
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[286:345, 322:389,:]
    height, width,_ = img.shape
    scale =4
    interpMethod = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_CUBIC,cv2.INTER_LANCZOS4]
    interpTitle = ['AREA','LINEAR','NEAREST','CUBIC','LANCZOS4']
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    for i in range(len(interpMethod)):
        plt.subplot(2, 3, i+2)
        imageResized = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=interpMethod[i])
        interpolationTitle = interpTitle[i]
        plt.imshow(imageResized)
        plt.title(interpTitle[i])
    plt.show()
if __name__ == "__main__":
    imageResize()