import cv2
import os
import numpy as np

def lucasKanade():
    root = os.getcwd()
    video_path = os.path.join(root, 'demoimages', 'MVI_0052.MP4')
    videoCapObj = cv2.VideoCapture(video_path)

    if not videoCapObj.isOpened():
        print("[ERROR] Unable to open video at:", video_path)
        return

    # Shi-Tomasi corner detection parameters
    shiTomasiCornerParams = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Lucas-Kanade optical flow parameters
    lucasKanadeParams = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Random colors for drawing tracks
    randomColors = np.random.randint(0, 255, (100, 3))

    # Read the first frame
    ret, firstFrame = videoCapObj.read()
    if not ret:
        print("[ERROR] Cannot read video frame.")
        return

    frameGrayPrev = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
    cornersPrev = cv2.goodFeaturesToTrack(frameGrayPrev, mask=None, **shiTomasiCornerParams)
    if cornersPrev is None:
        print("[ERROR] No features found in the first frame.")
        return

    mask = np.zeros_like(firstFrame)

    while True:
        ret, frame = videoCapObj.read()
        if not ret:
            break

        frameGrayCur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cornersCur, foundStatus, _ = cv2.calcOpticalFlowPyrLK(
            frameGrayPrev, frameGrayCur, cornersPrev, None, **lucasKanadeParams
        )

        if cornersCur is None or foundStatus is None:
            break

        cornersMatchedCur = cornersCur[foundStatus == 1]
        cornersMatchedPrev = cornersPrev[foundStatus == 1]

        for i, (cur, prev) in enumerate(zip(cornersMatchedCur, cornersMatchedPrev)):
            x_cur, y_cur = cur.ravel().astype(int)
            x_prev, y_prev = prev.ravel().astype(int)

            mask = cv2.line(mask, (x_cur, y_cur), (x_prev, y_prev), randomColors[i % 100].tolist(), 2)
            frame = cv2.circle(frame, (x_cur, y_cur), 5, randomColors[i % 100].tolist(), -1)

        output = cv2.add(frame, mask)
        cv2.imshow("Lucas-Kanade Optical Flow", output)

        if cv2.waitKey(30) & 0xFF == 27:  # Exit on pressing 'ESC'
            break

        frameGrayPrev = frameGrayCur.copy()
        cornersPrev = cornersMatchedCur.reshape(-1, 1, 2)

    videoCapObj.release()
    cv2.destroyAllWindows()
def denseOpticalFlow():
    root = os.getcwd()
    video_path = os.path.join(root, 'demoimages', 'MVI_0052.MP4')
    videoCapObj = cv2.VideoCapture(video_path)

    if not videoCapObj.isOpened():
        print("[ERROR] Unable to open video:", video_path)
        return

    ret, firstFrame = videoCapObj.read()
    if not ret:
        print("[ERROR] Unable to read first frame.")
        return

    imgPrev = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
    imgHSV = np.zeros_like(firstFrame)
    imgHSV[..., 1] = 255  # Set saturation to max

    while True:
        ret, frameCur = videoCapObj.read()
        if not ret:
            break

        imgCur = cv2.cvtColor(frameCur, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev=imgPrev, next=imgCur, flow=None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        imgHSV[..., 0] = angle * 180 / np.pi / 2  # Hue
        imgHSV[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value

        imgBGR = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)
        cv2.imshow("Dense Optical Flow", imgBGR)

        key = cv2.waitKey(15)
        if key == 27:  # ESC key
            break

        imgPrev = imgCur.copy()

    videoCapObj.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #lucasKanade()
    denseOpticalFlow()
