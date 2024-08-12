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
    hoopLeft, hoopRight = getHoopCoordinates(cap.read()[1])
    hoopMaxHeight = min(hoopLeft[1], hoopRight[1]) # pixels are counted from the top to the bottom so the max height is a lower y value
    hoopAverageHeight = (hoopLeft[1] + hoopRight[1]) / 2
    hoopMinHeight = max(hoopLeft[1], hoopRight[1])

    fga = 0
    fgm = 0

    cooldown = 0

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret:


            center, radius = findBasketballCenter(frame)
            cv2.circle(frame, hoopLeft, 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, hoopRight, 10, (0, 0, 255), cv2.FILLED)
            cv2.line(frame, (hoopLeft[0],hoopMinHeight), hoopRight, (0, 0, 255), 2)
            if center is not None:
                if center[1] <= (hoopMinHeight + radius * 4) and cooldown == 0:
                    posListX.append(center[0])
                    posListY.append(center[1])
                    if center[1] < hoopMinHeight:
                        shotInProgress = True
                        cv2.putText(frame, "Shot in Progress", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if len(posListX) > 3:
                if(posListY[-1] > hoopMinHeight and shotInProgress):
                    averageXOfLast2 = (posListX[-1] + posListX[-2]) / 2
                    if(hoopLeft[0] < averageXOfLast2 < hoopRight[0]):
                        fgm += 1
                        print("make")
                    fga += 1
                    print(f"FGM: {fgm}, FGA: {fga}, FG%: {100 * fgm/fga}")

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
def main():
    videoPath = 'NEAVid2.mov'
    traceContoursOnVideo(videoPath)


#PATH = r'E:\Youtube\tiktoks\footage\day 95\PXL_20231016_155547429.TS.mp4'
PATH = r"E:\Youtube\tiktoks\footage\10 freethrows\PXL_20240812_183026929.TS.mp4"
traceContoursOnVideo(PATH)

# i want to be able to determine if the ball is going to go in the hoop or not
# detect when a shot starts and when a new shot is taken
# record made and taken shots

