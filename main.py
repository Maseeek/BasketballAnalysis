import cv2
import numpy as np


def findBasketballCenter(frame, lastPosition):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredFrame = cv2.GaussianBlur(grayFrame, (9, 9), 0)
    edges = cv2.Canny(blurredFrame, 30, 100)
    cv2.imshow('edges', edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    radius = 0
    min_distance = float('inf')  # Initialize minimum distance as infinity

    # find the closest circle to the last position
    for contour in contours:
        (x, y), r = cv2.minEnclosingCircle(contour)
        area_contour = cv2.contourArea(contour)
        area_circle = np.pi * r * r

        # shape factor
        shape_factor = area_contour / area_circle

        # if the shape factor is close to 1, it's likely to be a circle
        if 0.8 < shape_factor < 1.2:
            if lastPosition is not None:  # Check if lastPosition is not None
                distance = np.sqrt((x - lastPosition[0])**2 + (y - lastPosition[1])**2)  # Calculate Euclidean distance
                if distance < min_distance:  # If this circle is closer to the last position
                    min_distance = distance
                    radius = r
                    center = (int(x), int(y))
            else:  # If lastPosition is None, select the contour based on the shape factor and radius
                if r > radius:
                    radius = r
                    center = (int(x), int(y))

    # draw the circle
    if center is not None:
        cv2.circle(frame, center, int(radius), (0, 255, 0), 2)

    return center, radius

image = cv2.imread('image5.jpg')
findBasketballCenter(image, None)



cap = cv2.VideoCapture('video.mp4')
lastPosition = None
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:

        center, radius = findBasketballCenter(frame, lastPosition)
        if(center is not None):
            lastPosition = center
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()