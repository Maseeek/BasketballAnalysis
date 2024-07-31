import cv2
import numpy as np


def findBasketballCenter(frame):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredFrame = cv2.GaussianBlur(grayFrame, (9, 9), 0)
    edges = cv2.Canny(blurredFrame, 30, 100)
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

image = cv2.imread('image5.jpg')
findBasketballCenter(image)



cap = cv2.VideoCapture('video.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        center, radius = findBasketballCenter(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()