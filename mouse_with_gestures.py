import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackModule as htm
import pyautogui

wCam, hCam = 640, 480
wScreen, hScreen = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetection(maxNumOfHands = 1, minDetConfidence=0.75,maxDetConfidence=0.75)

plocX, plocY = 0, 0
clocX, clocY = 0, 0
smooth = 10      #smoothening the trajectory of the point


pTime = 0
frame = 150     #size of the working area


while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw = False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUP()

        x3 = int(np.interp(x1, (frame, wCam - frame), (0, wScreen)))
        y3 = int(np.interp(y1, (frame, hCam - frame), (0, hScreen)))

        clocX = plocX + (x3 - plocX) / smooth
        clocY = plocY + (y3 - plocY) / smooth


        #print(fingers)
        if fingers[1] == 1 and fingers[2] == 0:         #moving mode
            cv2.circle(img, (x1, y1), 10, (0, 0, 139), cv2.FILLED)

            cv2.putText(img, 'Working area', (frame , frame ), cv2.FONT_ITALIC, 2, (0, 255, 0), 2 )
            cv2.rectangle(img, (frame, frame), (wCam - frame, hCam - frame), (0, 0, 255, 3))


            if fingers[0] == 1:
                pyautogui.click(clocX, clocY, duration = 0.11)      #tested on https://www.247checkers.com/

            else:
                pyautogui.moveTo(clocX, clocY, _pause=False)


        # if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0::
        #     #length,cx , cy, img = detector.distanceBtwinFingers(img, 8, 12)
        #     print("length")
        #     # if length < 40:
        #     #pyautogui.click(clocX, clocY, _pause=False)
        #     #pyautogui.dragTo(clocX, clocY, _pause=False, button='left')
        #     #cv2.circle(img, (clocX,cy), 5, (255, 0, 0), 3, cv2.FILLED)


        # if fingers[2] == 1 and fingers[3] == 1:
        #     pyautogui.dragTo(clocX, clocY, duration = 0.11, _pause=False, button = 'left' )
        #
        #     print("cos")


        plocX, plocY = clocX, clocY


    cTime = time.time()  # current time
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 20), cv2.FONT_ITALIC, 1, (0, 255, 0), 3)


    cv2.imshow("Img", img)
    cv2.waitKey(1)