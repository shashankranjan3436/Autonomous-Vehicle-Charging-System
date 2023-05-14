# AUTHOR: SHASHANK KUMAR RANJAN, IIT GUWAHATI
# CO-AUTHOR: KHUSHI AGARWAL, IIT GUWAHATI

import cv2 as cv
import numpy as np
import statistics
from matplotlib import pyplot as plt

img=cv.imread('C:/Users/SK RANJAN/Downloads/FFRGT.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, 0]], np.float32) 

# kernel = 1/3 * kernel

# gray = cv.filter2D(gray, -1, kernel)


def unsharp_mask(image, kernel_size=(3, 3), sigma=1.0, amount=2.0, threshold=500):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

gray = unsharp_mask(gray)

rows=gray.shape[0]

circles=cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,rows/14,param1=130,param2=25,minRadius=2,maxRadius=50)

# h = gray.shape[0]
# w = gray.shape[1]

# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(gray)
# top_left = min_loc
# bottom_right = (top_left[0] + w, top_left[1] + h)
# cv.rectangle(img,top_left, bottom_right, 255, 2)

x=[]
y=[]
rad=[]

if circles is not None:
    circles=np.uint16(np.around(circles))
    for i in circles[0,:]:
        center=(i[0],i[1])
        x.append(center[0]*(0.0265))
        y.append(center[1]*(0.0265))
        cv.circle(gray,center,1,(0,100,100),3)
        radius=i[2]
        rad.append(radius*(0.0265))
        cv.circle(gray,center,radius,(255,0,255),3)
        z=statistics.mode(rad)
print("x=",x)
print("y=",y)
print(rad)
print(z)
cv.imshow("detected",gray)
cv.waitKey(0)