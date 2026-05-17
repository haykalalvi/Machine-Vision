import cv2 as cv
import math

points = []

ratio = 10/137 #harus statis lokasi kameranya biar stabil nilai rationya

def draw_circle(event,x,y,flags,param):
    global points
    if event == cv.EVENT_LBUTTONDOWN:
        print(f"Klik terdeteksi di koordinat: x={x}, y={y}")
        if len(points) == 2:
            points = []
        points.append((x,y))
        # cv.circle(img,(x,y),100,(255,0,0),-1)


cv.namedWindow('frame')
cv.setMouseCallback('frame',draw_circle)



cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    for pt in points:
        cv.circle(frame,pt,20,(255,0,0),-1)


    if len(points) == 2:
        pt1 = points[0]
        pt2 = points[1]
        distance_px = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

        distance_cm = distance_px * ratio

        cv.putText(frame,text=f"{int(distance_cm)}",org=(pt1[0],pt1[1]-10),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(0,0,255),thickness=2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break