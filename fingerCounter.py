import cv2
import time
import HandTrackModule as htm




wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetection(minDetConfidence=0.75, maxDetConfidence=0.75)

tipId =[4, 8, 12, 16, 20]

while True:

    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0:
        fingersHigh = detector.fingersUP()
        #print(fingersUp)
        fingersUp = fingersHigh.count(1)
        cv2.putText(img, f'Number of fingers:{int(fingersUp)}', (10,20), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)


    cv2.imshow("image",img)
    cv2.waitKey(1)