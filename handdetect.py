import cv2
# import imutils
import numpy as np
import matplotlib.pyplot as plt

cap0 = cv2.VideoCapture(0)
# fgbg = cv2.bgsegm.createBackground
# cap1 = cv2.VideoCapture(0)

kernelD = np.ones((5, 5), np.uint8)
kernelE = np.ones((3, 3), np.uint8)
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([30, 255, 255], dtype = "uint8")

# lower = np.array([0, 80, 80], dtype = "uint8")
# upper = np.array([30, 255, 255], dtype = "uint8")
template = cv2.imread('./public/dataset/C_001.jpg', 0)


while(1):
    _, frame0 = cap0.read()
    edges = cv2.Canny(frame0, 100, 200)
    # hsv0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    # _, frame1 = cap1.read()
    # hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    

    hsv = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
    # gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # dila = cv2.dilate(dst, None)
    # frame0[ dst > 0.01*dst.max()] = [0,0,255]
    mask = cv2.inRange(hsv, lower, upper)
    dilation = cv2.dilate(mask, kernelD, iterations=2)
    erode = cv2.erode(mask, kernelE, iterations=1)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))

    hull = cv2.convexHull(contours)
    defects = cv2.convexityDefects(contours[0], hull)

    cv2.drawContours(frame0, [hull, contours], -1, (0,255,0), 2)


    cv2.imshow('frame', frame0)
    cv2.imshow('mask', mask)
    cv2.imshow('dilation', dilation)
    cv2.imshow('edges', edges)
    # cv2.imshow('erode', erode)
    # cv2.imshow('frame', frame0)
    cv2.imshow('hsv', hsv)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()