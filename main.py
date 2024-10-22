import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import math
#import matplotlib # import matplotlib.pyplot as plt doesnt work rn

dist = lambda x1, y1, x2, y2: (x1-x2)**2 + (y1-y2)**2
PUMPKIN = (33, 121, 250)
CELADON = (187, 229, 169)
VANILLA = (177, 246, 252)
FELDGRAU = (59, 75, 63)
GREEN = (63, 99, 68)
def get_video_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    return file_path
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
def drawHoop(frame, hoopLeft, hoopRight):
    cv2.circle(frame, hoopLeft, 10, GREEN, cv2.FILLED)
    cv2.circle(frame, hoopRight, 10, GREEN, cv2.FILLED)
    cv2.line(frame, hoopLeft, hoopRight, GREEN, 2)

def calculateAngle(positionListX, positionListY):
    # Calculate differences in coordinates
    try:
        delta_x = positionListX[2] - positionListX[0]
        delta_y = positionListY[2] - positionListY[0]

        # Calculate the angle in radians
        angle_radians = math.atan2(delta_y, delta_x)

        # Convert the angle to degrees
        angle_degrees = -math.degrees(angle_radians)
        #print(f"Angle in degrees: {angle_degrees}, Angle in radians: {angle_radians}, Delta X: {delta_x}, Delta Y: {delta_y}, positionListX: {positionListX[0:2]}, positionListY: {positionListY[0:2]}")
        if angle_degrees > 90 and angle_degrees < 180:
            angle_degrees = 180 - angle_degrees
        if(angle_degrees > 0 and angle_degrees < 90):
            return angle_degrees

        else:
            return 0
    except:
        return 0 # if there is an error or if angle is not within bounds, return 0
def calculateAverageAngle(shotAngles, shots):
    shotsMadeAngle = []
    shotsMissedAngle = []

    for i in range(len(shots)): # for each shot
        if (shotAngles[i] != 0): # as long as the shot angle of the shot is not 0
            if not abs(sum(shotAngles) / len(shotAngles) - shotAngles[i]) > 2 * sum(shotAngles) / len(shotAngles):
                if shots[i] == 1:
                    shotsMadeAngle.append(shotAngles[i])
                else:
                    shotsMissedAngle.append(shotAngles[i])
    try:
        averageAngle = (sum(shotsMadeAngle) + sum(shotsMissedAngle)) / (len(shotsMissedAngle) + len(shotsMadeAngle))
        if not len(shotsMadeAngle) == 0:
            averageMakeAngle = sum(shotsMadeAngle) / len(shotsMadeAngle)
        else:
            averageMakeAngle = 0
        if not len(shotsMissedAngle) == 0:
            averageMissAngle = sum(shotsMissedAngle) / len(shotsMissedAngle)
        else:
            averageMissAngle = 0


        return averageAngle, averageMakeAngle, averageMissAngle
    except:
        print("Error in calculating average angle")
        return 0, 0, 0
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

def showFrameWithBallCircled(frame, ball):
    if ball is not None:
        cv2.circle(frame, (ball[0], ball[1]), ball[2], PUMPKIN, 2)
        cv2.putText(frame, f"radius {ball[2]}", (ball[0], ball[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, PUMPKIN, 2, cv2.LINE_AA)
    cv2.waitKey(1)
def findBall(frame, prevCircle, radius):
    CONSTANT = 1.2
    minRadius = int (radius / CONSTANT)
    maxRadius = int (radius * CONSTANT)
    chosen = None
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)
    circles = cv2.HoughCircles(blurredFrame, cv2.HOUGH_GRADIENT, 1.2, 100,
                               param1=100, param2=30, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if(dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1])):
                    chosen = i
    return chosen
def tracePredictedPath(frame, posListX, posListY):
    A, B, C = np.polyfit(posListX, posListY, 2)
    widthOfFrame = frame.shape[1]
    xList = [item for item in range(widthOfFrame)]
    for i, (posX, posY) in enumerate(zip(posListX, posListY)):
        pos = (posX, posY)
        cv2.circle(frame, pos, 10, PUMPKIN, cv2.FILLED)
        if i == 0:
            cv2.line(frame, pos, pos, PUMPKIN, 5)
        else:
            cv2.line(frame, pos, (posListX[i - 1], posListY[i - 1]), PUMPKIN, 5)
    for x in xList:
        y = int(A * x ** 2 + B * x + C)
        cv2.circle(frame, (x, y), 2, FELDGRAU, cv2.FILLED) # draw the predicted path
    cv2.imshow('frame', frame)

def showResults(shots, shotAngles):
    # Count the number of makes and misses
    makes = shots.count(1)
    misses = shots.count(0)

    # Plot the pie chart with the correct labels
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Pie chart
    axs[0, 0].pie([makes, misses], labels=["Make", "Miss"], autopct='%1.1f%%')
    axs[0, 0].set_title('Field Goal Makes vs Misses')

    # Bar chart
    axs[0, 1].bar(['Makes', 'Misses'], [makes, misses], color=['green', 'red'])
    axs[0, 1].set_title('Shot Distribution')
    axs[0, 1].set_ylabel('Number of Shots')

    # Line chart for shot angles
    axs[1, 0].plot(shotAngles, marker='o')
    axs[1, 0].set_title('Shot Angles Over Time')
    axs[1, 0].set_xlabel('Shot Number')
    axs[1, 0].set_ylabel('Angle (degrees)')

    # Histogram for shot angles
    axs[1, 1].hist(shotAngles, bins=10, color='blue', edgecolor='black')
    axs[1, 1].set_title('Distribution of Shot Angles')
    axs[1, 1].set_xlabel('Angle (degrees)')
    axs[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Scatter plot for field goal percentage and shot angles
    fg_percentages = [100 * sum(shots[:i+1]) / (i+1) for i in range(len(shots))]
    plt.figure(figsize=(10, 7))
    plt.scatter(shotAngles, fg_percentages, c='blue', label='FG% vs Angle')
    plt.xlabel('Shot Angle (degrees)')
    plt.ylabel('Field Goal Percentage (%)')
    plt.title('Field Goal Percentage vs Shot Angle')
    plt.legend()
    plt.show()

    # Additional textual statistics
    longest_streak = getLongestStreak(shots)
    averageAngle, averageMakeAngle, averageMissAngle = calculateAverageAngle(shotAngles, shots)
    print(f"Longest Streak: {longest_streak}")
    print(f"Average Angle: {averageAngle:.2f}, Average Make Angle: {averageMakeAngle:.2f}, Average Miss Angle: {averageMissAngle:.2f}")
    print(f"FGM: {makes}, FGA: {len(shots)}, FG%: {100 * makes / len(shots):.2f}%")



def main(videoPath):

    # Initialize variables
    shots = []
    shotAngles = []
    posListX = []
    posListY = []
    cap = cv2.VideoCapture(videoPath)
    hoopLeft = getXandYValuesOfClick(cap.read()[1], "Left Side of Hoop")
    hoopRight = getXandYValuesOfClick(cap.read()[1], "Right Side of Hoop")
    ballRadius = 0.264 * math.sqrt(dist(hoopLeft[0], hoopLeft[1], hoopRight[0], hoopRight[1]))
    hoopMaxHeight = min(hoopLeft[1], hoopRight[1])
    hoopAverageHeight = (hoopLeft[1] + hoopRight[1]) / 2
    hoopMinHeight = max(hoopLeft[1], hoopRight[1])
    fga = 0
    fgm = 0
    cooldown = 0
    prevCircle = None
    center = None
    shotInProgress = False

    while cap.isOpened():
        ret, frame = cap.read()
        if fga != 0:
            cv2.putText(frame, f"FGM: {fgm}, FGA: {fga}, FG%: {(100 * fgm / fga):.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,1, VANILLA, 2, cv2.LINE_AA)

        if ret:
            basketball = findBall(frame, prevCircle, ballRadius)
            if basketball is not None:
                showFrameWithBallCircled(frame, basketball)
                prevCircle = (basketball[0], basketball[1], basketball[2])
                center = (int(basketball[0]), int(basketball[1]))
                radius = basketball[2]
            drawHoop(frame, hoopLeft, hoopRight)
            if center is not None:
                if center[1] <= (hoopMinHeight + radius * 5) and cooldown == 0:
                    posListX.append(center[0])
                    posListY.append(center[1])
                    if center[1] < hoopMinHeight:
                        shotInProgress = True
                        cv2.putText(frame, "Shot in Progress", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, VANILLA, 2,
                                    cv2.LINE_AA)

                        if len(posListX) > 1 and len(posListY) > 1:
                            angle = calculateAngle(posListX, posListY)
                            cv2.putText(frame, f"Release Angle: {angle:.2f}", (50, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, VANILLA, 2, cv2.LINE_AA)

            if len(posListX) > 3:
                if posListY[-1] > hoopMinHeight and shotInProgress:
                    averageXOfLast2 = (posListX[-1] + posListX[-2]) / 2
                    if hoopLeft[0] < averageXOfLast2 < hoopRight[0]:
                        shots.append(1)
                        fgm += 1
                    else:
                        shots.append(0)
                    fga += 1

                    shotAngles.append(calculateAngle(posListX, posListY))
                    posListX.clear()
                    posListY.clear()

                    shotInProgress = False
                    cooldown = 30
                else:
                    tracePredictedPath(frame, posListX, posListY)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # the delay between showing each frame, if 'q' is pressed, the video will close
                break
        else:
            break

        if cooldown > 0:
            cooldown -= 1

    cap.release()
    cv2.destroyAllWindows()
    #showResults(shots, shotAngles) # matplotlib.plyplot cannnot be imported

chooseVideo = True
PATH = r"E:\Youtube\tiktoks\footage\10 freethrows\PXL_20240812_183026929.TS.mp4"
if chooseVideo:
    PATH = get_video_path()
main(PATH)
shots = [1,0,1,1,0,1,0,0,0,1,1]
showResults(shots)

# make it work with a webcam - streaming device
# make it an executable file


#todays goals


