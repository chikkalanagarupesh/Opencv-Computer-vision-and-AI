import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def calibrateCamera(showPics=False):
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'demoimages')
    imagePathList = glob.glob(os.path.join(calibrationDir, '*82.jpg'))
    nrows = 9
    ncols = 14

    termcriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldptscurr = np.zeros((nrows * ncols, 3), np.float32)
    worldptscurr[:, :2] = np.mgrid[0:ncols, 0:nrows].T.reshape(-1, 2)

    worldptsList = []
    imgptsList = []

    for curImage in imagePathList:
        imgBGR = cv2.imread(curImage)
        if imgBGR is None:
            continue

        imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv2.findChessboardCorners(imgGray, (ncols, nrows), None)
        if cornersFound:
            worldptsList.append(worldptscurr)
            cornersRefined = cv2.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termcriteria)
            imgptsList.append(cornersRefined)
            if showPics:
                cv2.drawChessboardCorners(imgBGR, (ncols, nrows), cornersRefined, cornersFound)
                cv2.imshow("Chessboard Corners", imgBGR)
                cv2.waitKey(0)

    cv2.destroyAllWindows()

    if len(worldptsList) == 0 or len(imgptsList) == 0:
        print("[ERROR] No valid chessboard corners found.")
        return None, None

    retError, camMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        worldptsList, imgptsList, imgGray.shape[::-1], None, None)

    print('Camera Matrix:\n', camMatrix)
    print("Reprojection Error (pixels): {:.4f}".format(retError))

    paraPath = os.path.join(root, 'calibration.npz')
    np.savez(paraPath, camMatrix=camMatrix, distCoeffs=distCoeffs, rvecs=rvecs, tvecs=tvecs)
    
    return camMatrix, distCoeffs

def removeDistortion(camMatrix, distCoeff):
    root = os.getcwd()
    imgPath = os.path.join(root, 'demoimages', 'Prem_picture.JPG')
    img = cv2.imread(imgPath)

    if img is None:
        print("[ERROR] Could not load image for undistortion.")
        return

    height, width = img.shape[:2]
    camMatrixNew, roi = cv2.getOptimalNewCameraMatrix(camMatrix, distCoeff, (width, height), 1, (width, height))
    imgUndistorted = cv2.undistort(img, camMatrix, distCoeff, None, camMatrixNew)

    cv2.line(img, (1769, 103), (1780, 103), (255, 255, 255), 2)
    cv2.line(imgUndistorted, (1769, 103), (1780, 922), (255, 255, 255), 2)

    plt.figure()
    plt.subplot(121)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.title("Undistorted")
    plt.imshow(cv2.cvtColor(imgUndistorted, cv2.COLOR_BGR2RGB))
    plt.show()

def runCalibration():
    calibrateCamera(showPics=True)

def runRemoveDistortion():
    camMatrix, distCoeff = calibrateCamera(showPics=False)
    if camMatrix is not None and distCoeff is not None:
        removeDistortion(camMatrix, distCoeff)

if __name__ == '__main__':
    runCalibration()
    #runRemoveDistortion()
