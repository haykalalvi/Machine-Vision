import os
import cv2

# read image
image_path = os.path.join('.','data', 'bird.jpg')
img = cv2.imread(image_path)

# write
cv2.imwrite(os.path.join('.','data', 'bird_out.jpg'), img)

# visualize
cv2.imshow('image', img)
cv2.waitKey(0) # tell opencv to wait for a key press (jangan lupa) 0 tandanya ga akan di closo / 0ms
