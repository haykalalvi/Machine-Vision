import os
import cv2
import numpy as np


img = cv2.imread(os.path.join('.','data', 'basketball_player.jpg'))

img_edge = cv2.Canny(img, 50, 200) # trial and error aja angkanya

#fungsi tambahan buat melakukan dilatasi
img_edge_d = cv2.dilate(img_edge, np.ones((5,5), dtype=np.int8)) # make the line thicker
img_edge_e = cv2.erode(img_edge_d, np.ones((3,3), dtype=np.int8)) # make the line thinner

cv2.imshow("img", img)
cv2.imshow("img_edge", img_edge)
cv2.imshow("img_edge_d", img_edge_d)
cv2.imshow("img_edge_e", img_edge_e)
cv2.waitKey(0)