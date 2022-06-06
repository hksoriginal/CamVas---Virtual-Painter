
import handTrackingModule as htm
import time
import os
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import cv2
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)


root = Tk()
root.resizable(False, False)
root.geometry("1280x720")
root.title("CamVas - Virtual Painter")
root.iconbitmap('images/ico.ico')

#  main Label
bg_main = Image.open('images/bg.png')
bg_main = bg_main.resize((1280, 720), Image.ANTIALIAS)
bg_main = ImageTk.PhotoImage(bg_main)
main_frame = Label(root, image=bg_main)
main_frame.place(x=0, y=0, height=720, width=1280)


def paint():
    folderPath = "header"
    myList = os.listdir(folderPath)
    # print(myList)

    BLUE = (224, 169, 38)
    RED = (45, 30, 190)
    GREEN = (69, 147, 0)
    PURPLE = (255, 0, 255)
    BLACK = (0, 0, 0)
    drawColor = RED
    brushThickness = 15
    eraserThickness = 90

    overlayList = []

    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        image = cv2.resize(image, (1280, 125))
        overlayList.append(image)
    # print(len(overlayList))
    header = overlayList[0]
    # print("Printing Header", header)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    png = cv2.imread('header/1.png')
    # print(png.shape)

    detector = htm.handDetector(detectionCon=0.85)
    xp, yp = 0, 0

    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    # imgCanvas.fill(255)

    while True:
        # import image
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # find hand landmarks
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:

            # print(lmList)

            # tip of index and middle fingers
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # print(x1, y1, x2, y2)

            # check which finger is up
            fingers = detector.fingersUp()
            # print(fingers)

            # if selelection Mode - two fingers are up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                # print('Selection Mode')
                # checking for the click
                if y1 < 125:
                    if 250 < x1 < 450:
                        header = overlayList[0]
                        drawColor = RED
                    elif 550 < x1 < 750:
                        header = overlayList[1]
                        drawColor = BLUE
                    elif 800 < x1 < 950:
                        header = overlayList[2]
                        drawColor = GREEN
                    elif 1050 < x1 < 1200:
                        header = overlayList[3]
                        drawColor = BLACK
                cv2.rectangle(img, (x1, y1-25), (x2, y2+25),
                              drawColor, cv2.FILLED)

            # if drawing mode - index finger is up
            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                # print('Drawing Mode')
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == BLACK:
                    cv2.line(img, (xp, yp), (x1, y1),
                             drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1),
                             drawColor, eraserThickness)
                    xp, yp = x1, y1
                else:
                    cv2.line(img, (xp, yp), (x1, y1),
                             drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1),
                             drawColor, brushThickness)
                    xp, yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # setting the header image
        img[0:125, 0:1280] = header

        # cv2.imshow("image", img)
        # cv2.imshow("image Canvas", imgCanvas)

        # print(img.shape)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = ImageTk.PhotoImage(Image.fromarray(image))
        main_frame['image'] = img_array
        root.update()


# Draw Button
draw_btn = Image.open('images/draw.png')
draw_btn = draw_btn.resize((150, 50), Image.ANTIALIAS)
draw_btn = ImageTk.PhotoImage(draw_btn)
start_btn = Button(main_frame, command=paint, image=draw_btn, cursor="hand2",
                   relief='flat')
start_btn.place(x=600, y=660, width=150, height=50)


root.mainloop()
