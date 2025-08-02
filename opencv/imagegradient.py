import os
import cv2
import numpy as np  
import matplotlib.pyplot as plt

def imageGradient():
    root = os.getcwd()
    imgPath = os.path.join(root, 'demoimages//Prem_picture.JPG')
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("[ERROR] Image not found.")
        return

    plt.figure(figsize=(10, 8))
    
    # Original
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=21)
    plt.subplot(2, 2, 2)
    plt.imshow(laplacian, cmap='gray')  
    plt.title('Laplacian')
    plt.axis('off')

    # Sobel X
    kx, ky = cv2.getDerivKernels(1, 0, 3)
    print("Sobel X kernel:\n", ky @ kx.T)
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=21)
    plt.subplot(2, 2, 3)
    plt.imshow(sobelX, cmap='gray')
    plt.title('Sobel X')
    plt.axis('off')

    # Sobel Y
    kx, ky = cv2.getDerivKernels(0, 1, 3)
    print("Sobel Y kernel:\n", ky @ kx.T)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=21)
    plt.subplot(2, 2, 4)
    plt.imshow(sobelY, cmap='gray') 
    plt.title('Sobel Y')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    imageGradient()
