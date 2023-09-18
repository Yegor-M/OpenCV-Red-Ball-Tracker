import cv2
import sys
import numpy as np

img = cv2.imread('./red_ball.jpg')
if img is None:
    sys.exit("Could not read the image.")

img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower = np.array([160,100,20])
upper = np.array([179,255,255])

mask = cv2.inRange(img_hsv, lower, upper)

cv2.imshow("mask", mask)

kernel = np.ones((10, 10), np.uint8)

opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernel, iterations=1)
cv2.imshow("opening", opening)

closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
cv2.imshow("closing", closing)

segmented_img = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("segmented_img", segmented_img)

contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
cv2.imshow("output", output)

gray_image = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_image,255,255,255)
cv2.imshow("thresh", thresh)
cv2.imshow("gray_image", gray_image)

M = cv2.moments(thresh)

cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow("final image", img)

cv2.waitKey(0)