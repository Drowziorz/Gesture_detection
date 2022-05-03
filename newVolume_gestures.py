import cv2
import time
import numpy as np
import HandTrackModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480   #szerokosc i wysokosc okna


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

detector = htm.handDetection(maxDetConfidence=0.7, minDetConfidence=0.7, maxNumOfHands=1)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    area = 0

    if len(lmList) != 0:
        fingers = detector.fingersUP()
        area = (bbox[0] - bbox[2])*(bbox[1]-bbox[3]) // 100

        if 400<area<900:

            length, cx, cy, img = detector.distanceBtwinFingers(img,draw = True)
            #Setting up volume based on the position of thumb and index finger


            vol = np.interp(length,[20, 220], [0, 100])

            increment = 10
            vol = increment * round(vol/increment)

            if not fingers[3]:
                volume.SetMasterVolumeLevelScalar(vol/100, None)

                #volume bar
                RectHeight = np.interp(vol, [0, 100], [390, 150])
                cv2.rectangle(img, (50, int(RectHeight)), (150, 400), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, f'{int(vol)} %', (75, 450), cv2.FONT_ITALIC,
                            1, (255, 255, 0), 2)
                cv2.rectangle(img, (50, 150), (150, 400), (255, 0, 0), 3)



        else:
            cv2.putText(img, f'Set the hand properly', (75, 450), cv2.FONT_ITALIC,
                        1, (255, 255, 0), 2)


        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'fps:{int(fps)}', (10,30), cv2.FONT_ITALIC,
                    1, (255, 255, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)