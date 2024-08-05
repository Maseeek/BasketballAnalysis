import cv2
import numpy as np


def findBasketballCenter(frame):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)
    edges = cv2.Canny(blurredFrame, 30, 100)
    #cv2.imshow('edges', edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    radius = 0

    # find the largest circle
    for contour in contours:
        (x, y), r = cv2.minEnclosingCircle(contour)
        area_contour = cv2.contourArea(contour)
        area_circle = np.pi * r * r

        # shape factor
        shape_factor = area_contour / area_circle

        # if the shape factor is close to 1, it's likely to be a circle
        if r > radius and 0.8 < shape_factor < 1.2:
            radius = r
            center = (int(x), int(y))

    # draw the circle
    if center is not None:
        cv2.circle(frame, center, int(radius), (0, 255, 0), 2)

    return center, radius



def traceContoursOnVideo(videoPath):
    posListX = []
    posListY = []
    cap = cv2.VideoCapture(videoPath)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            center, radius = findBasketballCenter(frame)
            if center is not None:
                posListX.append(center[0])
                posListY.append(center[1])
                tracePredictedPath(frame, posListX, posListY)
            cv2.imshow('frame', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def tracePredictedPath(frame, posListX, posListY):
    A,B,C = np.polyfit(posListX, posListY, 2)
    widthOfFrame = frame.shape[1]
    xList = [item for item in range(widthOfFrame)]
    for i, (posX, posY) in enumerate(zip(posListX, posListY)):
        pos = (posX, posY)
        cv2.circle(frame, pos, 10, (0, 255, 0), cv2.FILLED)
        if i == 0:
            cv2.line(frame, pos, pos, (0, 255, 0), 5)
        else:
            cv2.line(frame, pos, (posListX[i - 1], posListY[i - 1]), (0, 255, 0), 5)
    for x in xList:
        y= int(A * x ** 2 + B * x + C)
        cv2.circle(frame, (x, y), 2, (255, 0, 255), cv2.FILLED)
    cv2.imshow('frame', frame)




#PATH = r'E:\Youtube\tiktoks\footage\day 95\PXL_20231016_155547429.TS.mp4'
PATH = 'NEAVid2.mov'
traceContoursOnVideo(PATH)

# i want to add a thing which looks at the last position of the ball
# using the last position of the ball it can choose the contour closest to it
# and then it can find the new position of the ball
# this way we can track the ball

