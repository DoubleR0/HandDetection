### Hand Presented ###

# import module
import cv2
import numpy as np
import math
import pyautogui
import time

# capture video
cap0 = cv2.VideoCapture(0)
# set range color
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([30, 255, 255], dtype = "uint8")
# set time counter
starttime = 0.0

while(1):
    # read frame
    _, frame0 = cap0.read()
    # set size of ROI
    roi=frame0[100:300, 100:300]
    # set position of rectangle
    cv2.rectangle(frame0,(100,100),(300,300),(0,255,0),2)
    
    # convert to hsv
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # covert range of color to black and white
    mask = cv2.inRange(hsv, lower, upper)
    # blur for delete noise
    blurred = cv2.blur(mask, (5,5))
    # set threshold hand
    ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)

    # find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # if no contours
    if(contours == []):
        # show frame
        cv2.imshow('frame0', frame0)
        # wait key of escape
        k = cv2.waitKey(33)
        if k == 27:
            break
    # if has contours
    else:
        # set lange of contours area
        contours = max(contours, key=lambda x: cv2.contourArea(x))
        # draw contours on frame
        cv2.drawContours(roi, [contours], -1, (255,255,0), 2)

        # return convexHull and find convexityDefects
        hull = cv2.convexHull(contours, returnPoints=False)
        defects = cv2.convexityDefects(contours, hull)

        # set defects counter
        count_defects = 0
        # loop find defects
        for i in range(defects.shape[0]):
            # print("defects ===== ", defects[i, 0])
            # set position of defects
            s, e, f, d = defects[i, 0]
            # separate start, end and far
            start = tuple(contours[s][0])
            end = tuple(contours[e][0])
            far = tuple(contours[f][0])

            # clculate triangle find angle less than 90
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # if angle < 90 draw a circle at the end point
            if angle <= 90:
                count_defects += 1
                cv2.circle(roi, end, 10, [0, 0, 255], -1)
            # draw convexHull in ROI
            cv2.line(roi, start, end, [0, 255, 255], 2)

        # action each number of defect
        if count_defects == 1:
            # set endtime
            endtime = time.time()
            if (endtime-starttime) > 2:
                # action keybord event
                pyautogui.press('right')
                print("right")
                # set new starttime
                starttime = time.time()
            else:
                print("wait")
        elif count_defects == 3:
            # set endtime
            endtime = time.time()
            if (endtime-starttime) > 2:
                # action keybord event
                pyautogui.press('left')
                print("left")
                # set new starttime
                starttime = time.time()
            else:
                print("wait")
            #  time.sleep(2)
        elif count_defects == 4:
            # action keybord event
            pyautogui.press('esc')

        # show frame
        cv2.imshow('frame0', frame0)
        # wait key of escape
        k = cv2.waitKey(33)
        if k == 27:
            break
cv2.destroyAllWindows()