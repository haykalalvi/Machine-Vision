import os
import cv2

img = cv2.imread(os.path.join('.','data', 'dogs.jpg'))
print(img.shape) # di terminal height dan width
crop_img = img[100:200, 100:200] # height dulu

cv2.imshow("img", img)
cv2.imshow("crop_img", crop_img)
cv2.waitKey(0)