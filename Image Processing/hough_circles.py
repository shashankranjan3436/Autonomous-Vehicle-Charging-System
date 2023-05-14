# AUTHOR: SHASHANK KUMAR RANJAN, IIT GUWAHATI
# CO-AUTHOR: KHUSHI AGARWAL, IIT GUWAHATI

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img=cv.imread('C:/Users/SK RANJAN/Downloads/FFRGT.jpg')

# img=cv.imread('/Users/khushi/Downloads/FFRGT.jpg')

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

h = gray.shape[0]
w = gray.shape[1]

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(gray)
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(img,top_left, bottom_right, 255, 2)



if circles is not None:
    circles=np.uint16(np.around(circles))
    for i in circles[0,:]:
        center=(i[0],i[1])
        cv.circle(gray,center,1,(0,100,100),3)
        radius=i[2]
        cv.circle(gray,center,radius,(255,0,255),3)

plt.subplot(121),plt.imshow(gray,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

cv.imshow("detected",gray)
cv.waitKey(0)




# import cv2
# import numpy as np

# # Load image, grayscale, median blur, Otsus threshold
# image = cv2.imread('/Users/khushi/Desktop/ttryy.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.medianBlur(gray, 11)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# # Morph open 
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

# # Find contours and filter using contour area and aspect ratio
# cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.04 * peri, True)
#     area = cv2.contourArea(c)
#     if len(approx) > 5 and area > 1000 and area < 500000:
#         ((x, y), r) = cv2.minEnclosingCircle(c)
#         cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)

# cv2.imshow('thresh', thresh)
# cv2.imshow('opening', opening)
# cv2.imshow('image', image)
# cv2.waitKey() 