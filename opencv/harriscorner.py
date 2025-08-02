import os
import cv2
import numpy as np      
import matplotlib.pyplot as plt

def harrisCorner():
    root = os.getcwd()
    imgPath = os.path.join(root, 'demoimages//Prem_picture.JPG')
    img = cv2.imread(imgPath)

    if img is None:
        print("[ERROR] Image not found.")
        return

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    imgGray = np.float32(imgGray)

    plt.figure()

    plt.subplot(1, 3, 1)
    plt.imshow(imgRGB)  # No cmap for RGB
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    blocksize = 5
    sobelsize = 3
    k = 0.04
    harris = cv2.cornerHarris(imgGray, blocksize, sobelsize, k)
    plt.imshow(harris, cmap='gray')
    plt.title("Harris Response")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    imgRGB[harris > 0.05 * harris.max()] = [255, 0, 0]
    plt.imshow(imgRGB)
    plt.title("Corners Marked")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    harrisCorner()
