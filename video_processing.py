import numpy as np
import cv2 as cv
import imageio

capture = cv.VideoCapture('./rgb_ball_720.mp4')
frames = []
frames = frames[::4]
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        print("Stream is empty.")
        break

    img_hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    lower_red = np.array([0, 170, 100])
    upper_red = np.array([10, 255, 255])

    mask_red = cv.inRange(img_hsv, lower_red, upper_red)

    kernel = np.ones((12,12),np.uint8)
    opening_mask = cv.morphologyEx(mask_red, cv.MORPH_CLOSE, kernel)
    closing_mask = cv.morphologyEx(opening_mask, cv.MORPH_OPEN, kernel)

    segmented_img = cv.bitwise_and(frame, frame, mask=closing_mask)
    cv.imshow("segmented_img", segmented_img)

    contours, hierarchy = cv.findContours(closing_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    output = cv.drawContours(frame, contours, -1, (255, 255, 230), 3)

    gray_image = cv.cvtColor(segmented_img, cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(gray_image,255,255,255)

    M = cv.moments(thresh)
    if(M["m10"] != 0.0):
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
        cv.putText(frame, "centroid", (cX - 25, cY - 25),cv.QT_FONT_NORMAL, 0.5, (255, 255, 255), 2)

    frames.append(frame)
    cv.imshow('frame, click q to quit', frame)

    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()

imageio.mimsave('output.gif', frames, duration=0.05)