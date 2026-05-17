import numpy as np
import cv2

# def get_limits(color):
#     c = np.uint8([[color]])
#     hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    
#     lowerLimit = hsvC[0][0][0] - 10, 100, 100
#     upperLimit = hsvC[0][0][0] + 10, 255, 255

#     lowerLimit = np.array(lowerLimit, dtype=np.uint8)
#     upperLimit = np.array(upperLimit, dtype=np.uint8)

#     return lowerLimit, upperLimit
    



import numpy as np
import cv2

def get_limits(color):
    # Buat pixel 1x1 dari warna BGR dan convert ke HSV
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    
    # Ambil nilai Hue-nya saja
    hue = hsvC[0][0][0]
    
    # Hitung batas atas dan bawah dengan aman (tidak boleh kurang dari 0 atau lebih dari 179)
    # Kita turunkan batas S dan V menjadi 50 agar lebih toleran terhadap cahaya redup
    hue_lower = max(0, int(hue) - 10)
    hue_upper = min(179, int(hue) + 10)

    lowerLimit = np.array([hue_lower, 50, 50], dtype=np.uint8)
    upperLimit = np.array([hue_upper, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit