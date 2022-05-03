import cv2
import mediapipe as mp
import time
import math

class handDetection():
    def __init__(self,
                 mode = False,
                 maxNumOfHands = 2,
                 modelComplexity = 1,
                 minDetConfidence = 0.5,
                 maxDetConfidence = 0.5):
        self.mode = mode
        self.maxNumOfHands = maxNumOfHands
        self.modelComplexity = modelComplexity
        self.minDetConfidence = minDetConfidence
        self.maxDetConfidence = maxDetConfidence

        self.mpHands = mp.solutions.hands                           #inicjalizacja
        self.hands = self.mpHands.Hands(self.mode, self.maxNumOfHands, self.modelComplexity,
                                        self.minDetConfidence, self.maxDetConfidence )
        self.mpDraw = mp.solutions.drawing_utils

        self.tipId = [4, 8, 12, 16, 20]



    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:  # jesli zostanie wykryta dlon
            for handLms in self.results.multi_hand_landmarks:  # rysowanie dla kazdej z dloni
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def distanceBtwinFingers(self, img,firstFinger = 4, secondFinger = 8, draw = True):

        x1, y1 = self.lmList[firstFinger][1], self.lmList[firstFinger][2]
        x2, y2 = self.lmList[secondFinger][1], self.lmList[secondFinger][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot((x1 - x2), (y1 - y2))
        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)
            # if length > 180:
            #     cv2.circle(img, (cx, cy), 10, (0, 83, 229), cv2.FILLED)


        return length, cx, cy, img

    def findPosition(self, img, HandNumber = 0, draw = True):
        xlist, ylist = [], []
        self.lmList = []
        bbox = []
        if self.results.multi_hand_landmarks:
            currentHand = self.results.multi_hand_landmarks[HandNumber]

            for id, lm in enumerate(currentHand.landmark):        #wyswietlanie pozycji w konsoli
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                xlist.append(cx)
                ylist.append(cy)
                if draw:
                       cv2.circle(img, (cx,cy), 10, (255, 125, 255), cv2.FILLED)

        if len(xlist) != 0:
            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[2],bbox[3]), (0, 0, 255), 3)

        return self.lmList, bbox

    def fingersUP(self):
        fingers = []

        if (self.lmList[self.tipId[0]][1] < self.lmList[self.tipId[0] - 1][1]):  # only for thumb
            fingers.append(0)
        else:
            fingers.append(1)

        for id in range(1, len(self.tipId)):
            if (self.lmList[self.tipId[id]][2] < self.lmList[self.tipId[id] - 2][2]):
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers



def main():
    pTime = 0
    cap = cv2.VideoCapture(0)

    detector = handDetection()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw = False)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()     # current time
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 20), cv2.FONT_ITALIC, 1, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()