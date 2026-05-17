import os
import cv2

# read image
image = os.path.join('.','data', 'freelancer.jpg')
img = cv2.imread(image)

k_size = 15
img_blur = cv2.blur(img, (k_size,k_size))
img_gaussian = cv2.GaussianBlur(img, (k_size,k_size),3)
img_median = cv2.medianBlur(img, k_size)
# visualize
cv2.imshow('image', img)
cv2.imshow('image_blur', img_blur)
cv2.imshow('image_gaussian', img_gaussian)
cv2.imshow('image_median', img_median)
cv2.waitKey(0) # tell opencv to wait for a key press (jangan lupa) 0 tandanya ga akan di closo / 0ms
