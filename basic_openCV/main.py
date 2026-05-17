import cv2
from PIL import Image # dia ini kayak image editing, dipake sekadar untuk bbox aja
from basic_openCV.util import get_limits

# we run in local with cpu not gpu

color = [255, 0, 0] # INI DI bgr
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    """
    So, karena ini mau deteksi color, biasanya kita convert ke HSV colorspaces lalu kita tinjau dari Hue regionnya
    Hue regionnya itu kan spektrum ya, jadi kita define nilai rentangnya dulu biar nanti pythonnya tinggal
    ninjau nilai pixel di rentang yang udah kita define sebelumnya
    
    
    """

    hsvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=color)

    mask = cv2.inRange(hsvimage, lowerLimit, upperLimit) # ngubah jadi warna putih untuk warna yang terdeteksi
    #convert numpy array to pillow
    mask_ = Image.fromarray(mask) # dari format matriks numpy array opencv, diubah jadi objek gambar pillow

    bbox = mask_.getbbox()

    print(bbox)


    if bbox is not None:
        x1, y1, x2, y2 = bbox

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)    


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()