# AUTHOR: SHASHANK KUMAR RANJAN, IIT GUWAHATI
# CO-AUTHOR: KHUSHI AGARWAL, IIT GUWAHATI

import cv2 as cv
import random
import numpy as np

img = cv.imread('C:/Users/SK RANJAN/Downloads/FFRGT.jpg')

h = img.shape[0]
w = img.shape[1]

def fill(img, h, w):
    img = cv.resize(img, (h,w), cv.INTER_CUBIC)
    return img

def hor_shift(img, r, h, w):
    s = w*r
    if r>0:
        img = img[:, :int(w-s), :]
    else:
        img = img[:, :int(-1*s):, :]
    img = fill(img, h, w)
    return img

def ver_shift(img, r, h, w):
    s = h*r
    if r>0:
        img = img[:int(h-s), :, :]
    else:
        img = img[int(-1*s):, :, :]
    img = fill(img, h, w)
    return img

def bright(img, low, high):
    val = random.uniform(low, high)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*val
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*val 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint)
    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return img

def zoom(img, val, h, w):
    h_taken = int(val*h)
    w_taken = int(val*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken]
    img = fill(img, h, w)
    return img

def rotation(img, val, h, w):
    val = int(random.uniform(-val, val))
    M = cv.getRotationMatrix2D((int(w/2), int(h/2)), val, 1)
    img = cv.warpAffine(img, M, (w, h))
    return img

img_shifted_hor = hor_shift(img, 0.6, h, w)
img_shifted_ver = hor_shift(img, 0.6, h, w)
img_bright = bright(img, 0.5, 3)
img_zoom = zoom(img, 0.5, h, w)
img_rotate = rotation(img, 30, h,w)

cv.imwrite('Shifted_h.png', img_shifted_hor)
cv.imwrite('Shifted_v.png', img_shifted_ver)
cv.imwrite('Brightness.png', img_bright)
cv.imwrite('Zoom.png', img_zoom)
cv.imwrite('Rotated.png', img_rotate)

cv.waitKey(0)
cv.destroyAllWindows()