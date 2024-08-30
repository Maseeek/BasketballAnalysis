import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import math


def findBasketballCenter(frame):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)
    edges = cv2.Canny(blurredFrame, 30, 100)
    # cv2.imshow('edges', edges)
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

def calculateAngle(positionListX, positionListY):
    # Calculate differences in coordinates
    try:
        delta_x = positionListX[1] - positionListX[0]
        delta_y = positionListY[1] - positionListY[0]

        # Calculate the angle in radians
        angle_radians = math.atan2(delta_y, delta_x)

        # Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)
        return -angle_degrees

    except:
        return 0

    return -angle_degrees

def get_video_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    return file_path


def drawHoop(frame, hoopLeft, hoopRight):
    cv2.circle(frame, hoopLeft, 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(frame, hoopRight, 10, (0, 0, 255), cv2.FILLED)
    cv2.line(frame, hoopLeft, hoopRight, (0, 0, 255), 2)


def main(videoPath):
    shots = []
    shotAngles = []
    posListX = []
    posListY = []
    cap = cv2.VideoCapture(videoPath)
    hoopLeft, hoopRight = getHoopCoordinates(cap.read()[1])
    hoopMaxHeight = min(hoopLeft[1], hoopRight[1])  # pixels are counted from the top to the bottom so the max height is a lower y value
    hoopAverageHeight = (hoopLeft[1] + hoopRight[1]) / 2
    hoopMinHeight = max(hoopLeft[1], hoopRight[1])

    fga = 0
    fgm = 0

    cooldown = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if (fga != 0):
            cv2.putText(frame, f"FGM: {fgm}, FGA: {fga}, FG%: {100 * fgm / fga}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        if ret:

            center, radius = findBasketballCenter(frame)
            drawHoop(frame, hoopLeft, hoopRight)
            if center is not None:
                if center[1] <= (hoopMinHeight + radius * 4) and cooldown == 0:
                    posListX.append(center[0])
                    posListY.append(center[1])
                    if center[1] < hoopMinHeight:
                        shotInProgress = True
                        cv2.putText(frame, "Shot in Progress", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                    cv2.LINE_AA)

                        cv2.putText(frame, f"Release Angle: {calculateAngle(posListX, posListY)}", (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if len(posListX) > 3:
                if (posListY[-1] > hoopMinHeight and shotInProgress):
                    averageXOfLast2 = (posListX[-1] + posListX[-2]) / 2
                    if (hoopLeft[0] < averageXOfLast2 < hoopRight[0]):
                        shots.append(1)
                        fgm += 1
                        #print("make")
                    else:
                        shots.append(0)
                    fga += 1


                    #print(f"FGM: {fgm}, FGA: {fga}, FG%: {100 * fgm / fga}")

                    shotAngles.append(calculateAngle(posListX, posListY))
                    posListX.clear()
                    posListY.clear()

                    shotInProgress = False
                    cooldown = 30
                else:
                    tracePredictedPath(frame, posListX, posListY)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        if cooldown > 0:
            cooldown -= 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Longest Streak: {getLongestStreak(shots)}")
    averageAngle, averageMakeAngle, averageMissAngle = calculateAverageAngle(shotAngles, shots)
    print(f"Average Angle: {averageAngle}, Average Make Angle: {averageMakeAngle}, Average Miss Angle: {averageMissAngle}")
    print(shotAngles)

def calculateAverageAngle(shotAngles, shots):
    shotsMadeAngle = []
    shotsMissedAngle = []

    for i in range(len(shots)):
        if shots[i] == 1:
            shotsMadeAngle.append(shotAngles[i])
        else:
            shotsMissedAngle.append(shotAngles[i])
    try:
        averageMakeAngle = sum(shotsMadeAngle) / len(shotsMadeAngle)
        averageMissAngle = sum(shotsMissedAngle) / len(shotsMissedAngle)
        averageAngle = sum(shotAngles) / (len(shotsMissedAngle)+len(shotsMadeAngle))
        return averageAngle, averageMakeAngle, averageMissAngle
    except:
        return 0, 0, 0
def tracePredictedPath(frame, posListX, posListY):
    A, B, C = np.polyfit(posListX, posListY, 2)
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
        y = int(A * x ** 2 + B * x + C)
        cv2.circle(frame, (x, y), 2, (255, 0, 255), cv2.FILLED)
    cv2.imshow('frame', frame)

def getLongestStreak(array):
    longestStreak = 0
    currentStreak = 0
    for i in range(len(array)):
        if array[i] == 1:
            currentStreak += 1
            if currentStreak > longestStreak:
                longestStreak = currentStreak
        else:
            currentStreak = 0
    return longestStreak
def getXandYValuesOfClick(frame, windowName):
    # function to detect mouse clicks
    def mouseClick(event, x, y, flags, param):
        nonlocal click_x, click_y, clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            click_x, click_y = x, y
            clicked = True

    clicked = False
    click_x, click_y = -1, -1

    # creates a window for user to click
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, mouseClick)

    # displays image
    cv2.imshow(windowName, frame)

    while not clicked:
        cv2.waitKey(1)  # waits until the screen is clicked

    cv2.destroyAllWindows()

    return click_x, click_y


def getHoopCoordinates(frame):
    hoopLeft = getXandYValuesOfClick(frame, "Left Side of Hoop")
    hoopRight = getXandYValuesOfClick(frame, "Right Side of Hoop")
    return hoopLeft, hoopRight


chooseVideo = False
PATH = r"E:\Youtube\tiktoks\footage\10 freethrows\PXL_20240812_183026929.TS.mp4"
if chooseVideo:
    PATH = get_video_path()

main(PATH)

#modularise the code
#get angle of the shot
