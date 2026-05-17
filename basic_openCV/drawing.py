import os
import cv2


img = cv2.imread(os.path.join('.','data', 'whiteboard.jpg'))

#line | starting coor, end coor, color, thickness
cv2.line(img,(200,250),(300,300),(0,255,0), 3)

# rectangle | starting coor, end coor, color, thickness
cv2.rectangle(img,(200,250),(300,300),(0,0,255), 3)

# circle | center coor, radius, color, thickness
cv2.circle(img,(200,250),50,(255,0,0), 3)

# text |    text, starting coor, font, font scale, color, thickness
cv2.putText(img, "Hello", (200,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

cv2.imshow("img", img)
cv2.waitKey(0)