import os
import cv2
import numpy as np      
import matplotlib.pyplot as plt
def SIFT():
    root = os.getcwd()
    imgPath = os.path.join(root, 'demoimages//Prem_picture.JPG')
    imgGray = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    if imgGray is None:
        print("[ERROR] Image not found.",imgPath)
        return
    sift = cv2.SIFT_create()
    keypoints= sift.detect(imgGray, None)
    img_with_keypoints= cv2.drawKeypoints(imgGray,keypoints,imgGray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure()
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title("SIFT Keypoints")
    plt.show()

if __name__ == "__main__":
    SIFT()
