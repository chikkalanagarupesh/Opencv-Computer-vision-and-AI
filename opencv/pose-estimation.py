import cv2
import os
import numpy as np
import enum
import glob

class DrawOption(enum.Enum):
    AXES = 1
    CUBE = 2

def drawAxes(img, corners, imgpts):
    def tupleOfInts(arr):
        return tuple(int(x) for x in arr)

    corner = tupleOfInts(corners[0].ravel())
    img = cv2.line(img, corner, tupleOfInts(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tupleOfInts(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tupleOfInts(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

def drawCube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    for i in range(4):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i + 4]), (255, 0, 0), 3)

    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

def poseEstimation(option: DrawOption):
    root = os.getcwd()
    paramPath = os.path.join(root, 'demoimages/calibration_file/calibration.npz')

    data = np.load(paramPath)
    camMatrix = data['camMatrix']
    distCoeff = data['distCoeffs']

    calibrationDir = os.path.join(root, 'demoimages/10104782')
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    nRows = 9
    nCols = 14

    termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtscur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtscur[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]])
    cubeCorners = np.float32([
        [0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
        [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]
    ])

    for curImgPath in imgPathList:
        imgBGR = cv2.imread(curImgPath)
        if imgBGR is None:
            print(f"Image not found: {curImgPath}")
            continue

        imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        cornersFound, cornersorg = cv2.findChessboardCorners(imgGray, (nCols, nRows), None)

        if cornersFound:
            cornersRefined = cv2.cornerSubPix(imgGray, cornersorg, (11, 11), (-1, -1), termCriteria)
            cornersRefined = cornersRefined.reshape(-1,2)
            print("worldPtscur shape:",worldPtscur.shape)
            print("cornersRefined shape:",cornersRefined.shape)
            success, rvecs, tvecs = cv2.solvePnP(worldPtscur, cornersRefined, camMatrix, distCoeff)

            if not success:
                print(f"Pose estimation failed for image: {curImgPath}")
                continue

            if option == DrawOption.AXES:
                imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, camMatrix, distCoeff)
                imgBGR = drawAxes(imgBGR, cornersRefined, imgpts)
            elif option == DrawOption.CUBE:
                imgpts, _ = cv2.projectPoints(cubeCorners, rvecs, tvecs, camMatrix, distCoeff)
                imgBGR = drawCube(imgBGR, imgpts)

            cv2.imshow('Chessboard', imgBGR)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    poseEstimation(DrawOption.AXES)    