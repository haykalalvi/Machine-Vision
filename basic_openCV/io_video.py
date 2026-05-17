import os
import cv2

# read_video
video_path = os.path.join('.','data', 'video.mp4')
video = cv2.VideoCapture(video_path)


#visualize video
ret = True # dia boolean buat ngecek masih ada frame yang terbaca ga
while ret:
    ret, frame = video.read()

    if ret: # dia nampilin video per framenya
        cv2.imshow('frame', frame)
        cv2.waitKey(40) # didapet dari video yang memiliki 25 frame per detik

# kedua ini buat free up space memory yang dipake buat nampilin videonya
video.release()
cv2.destroyAllWindows()
