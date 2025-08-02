import cv2
import os

def grayscale():
    root = os.getcwd()
    filename = 'demoimages//LU1H9206.JPG'
    imgPath = os.path.join(root, filename)
    img = cv2.imread(imgPath) 
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray_resized = cv2.resize(imgGray, (320, 240))
    
    cv2.imshow('gray resized', imgGray_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def readAsGray():
    root = os.getcwd()
    filename = 'demoimages//LU1H9206.JPG'
    imgPath = os.path.join(root, filename)
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    imgGray_resized = cv2.resize(img, (320, 240))
    
    cv2.imshow('gray resized', imgGray_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #grayscale()
    readAsGray()

