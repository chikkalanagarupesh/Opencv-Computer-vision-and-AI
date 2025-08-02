import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class DepthMap:
    def __init__(self, showImages):
        root = os.getcwd()
        imgLeftPath = os.path.join(root, 'demoImages/motorcycle.png')
        imgRightPath = os.path.join(root, 'demoImages/motorcycle 1.png')

        self.imgLeft = cv2.imread(imgLeftPath, cv2.IMREAD_GRAYSCALE)
        self.imgRight = cv2.imread(imgRightPath, cv2.IMREAD_GRAYSCALE)

        if self.imgLeft is None or self.imgRight is None:
            raise FileNotFoundError("Could not load left or right image.")

        if showImages:
            plt.figure()
            plt.subplot(121)
            plt.title("Left Image")
            plt.imshow(self.imgLeft, cmap='gray')
            plt.subplot(122)
            plt.title("Right Image")
            plt.imshow(self.imgRight, cmap='gray')
            plt.show()

    def computeDepthMapBM(self):
        nDispFactor = 6  
        stereo = cv2.StereoBM_create(numDisparities=16 * nDispFactor, blockSize=21)
        disparity = stereo.compute(self.imgLeft, self.imgRight)
        plt.imshow(disparity, cmap='gray')
        plt.title("Depth Map - StereoBM")
        plt.colorbar()
        plt.show()

    def computeDepthMapSGBM(self):
        window_size = 7
        min_disp = 16
        nDispFactor = 14 
        num_disp = 16 * nDispFactor - min_disp

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disparity = stereo.compute(self.imgLeft, self.imgRight).astype(np.float32) / 16.0
        plt.imshow(disparity, cmap='gray')
        plt.title("Depth Map - StereoSGBM")
        plt.colorbar()
        plt.show()

def demoViewPics():
    dp = DepthMap(showImages=True)

def demoStereoBM():
    dp = DepthMap(showImages=False)
    dp.computeDepthMapBM()

def demoStereoSGBM():
    dp = DepthMap(showImages=False)
    dp.computeDepthMapSGBM()

if __name__ == '__main__':
    demoViewPics()
    # Uncomment the lines below to run BM or SGBM
    #demoStereoBM()
    # demoStereoSGBM()
