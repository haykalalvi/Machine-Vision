import os

import cv2

img = cv2.imread(os.path.join('.','data', 'bear.jpg.webp'))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# namanya glbal threshold, ada satu nilai yan dipake sebagai thresholdnya
ret, thresh = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY) # yang di bawah 80 jadi 0 (black)

cv2.imshow("img", img)
cv2.imshow("img_gray", img_gray)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)