import cv2
import os
import numpy as np
import matplotlib.pyplot as plt 

def callback(input):
    pass    

def gaussianKernel(size, sigma):
    kernel = cv2.getGaussianKernel(size, sigma)
    kernel = np.outer(kernel, kernel)
    return kernel   

def gaussianFiltering():
    root = os.getcwd()
    filename = 'demoimages//Prem_picture.JPG'
    imgPath = os.path.join(root, filename)
    img = cv2.imread(imgPath)

    # Resize image
    height, width, _ = img.shape
    scale = 1 / 4
    width = int(width * scale)
    height = int(height * scale)
    img = cv2.resize(img, (width, height))

    # Show the kernel in matplotlib before cv2 GUI
    n = 51
    kernel = gaussianKernel(n, 8)
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(kernel, cmap='gray')
    ax = fig.add_subplot(122, projection='3d')
    x = np.arange(0, n, 1)
    y = np.arange(0, n, 1)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, kernel, cmap='viridis')
    plt.show()  # This blocks, so we do it BEFORE GUI stuff

    # Now OpenCV window and trackbar
    winName = 'Gaussian Filter'
    cv2.namedWindow(winName)
    cv2.createTrackbar('sigma', winName, 1, 20, callback)

    while True:
        sigma = cv2.getTrackbarPos('sigma', winName)
        if sigma < 1:
            sigma = 1
        imgFilter = cv2.GaussianBlur(img, (n, n), sigma)
        cv2.imshow(winName, imgFilter)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    gaussianFiltering()

