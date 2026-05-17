import cv2

# read webcam
webcam = cv2.VideoCapture(0) # tulis berapa webcam yang mau lu akses

# visualize webcam
while True:
    ret,frame = webcam.read()

    cv2.imshow('frame', frame)
    # waktunya sebenernya ga 40ms gitu, kalau pake webcam lebih complex
    if cv2.waitKey(40) & 0xFF == ord('q'): # ini buat terminate
        break



webcam.release()
cv2.destroyAllWindows()