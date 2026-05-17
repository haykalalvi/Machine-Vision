import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os


output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# read image 
#img_path = './data/dogs.jpg'
#img = cv2.imread(os.path.join('.','data', 'dogs.jpg'))


#img = cv2.imread(os.path.join('.','data','bird.jpg'))
#img = cv2.imread(img_path)
# cv2.imshow('image', img)
# cv2.waitKey(0) # tell opencv to wait for a key press (jangan lupa) 0 tandanya ga akan di closo / 0ms


# Load the input image from an image file.
mp_image = mp.Image.create_from_file('./data/person.jpg')
#print(img.shape) # di terminal height dan width

H, W = mp_image.height, mp_image.width


# Zoom in (scale up by 2x)
# zoom_factor = 2
# resized_img = cv2.resize(mp_image, (1944,2592)) # width then height
# cv2.imshow("img_hsv", resized_img)
# cv2.waitKey(0)

# Load the input image from a numpy array.
#mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    
# detect face

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the image mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='./data/blaze_face_full_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)


with FaceDetector.create_from_options(options) as detector:
    # Lakukan deteksi wajah
    face_detector_result = detector.detect(mp_image)
    
     # --- TAMBAHKAN KODE INI UNTUK MELIHAT HASILNYA ---
    detections = face_detector_result.detections
    
    # ... (kode setup detector di atas tetap sama) ...
    
    if not detections:
        print("Tidak ada wajah yang terdeteksi.")
    else:
        print(f"Berhasil! Ditemukan {len(detections)} wajah.")
        
        # 1. KONVERSI GAMBAR UNTUK OPENCV
        # Ekstrak numpy array dari mp_image dan ubah warnanya kembali ke BGR
        img_bgr = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)
        
        # Looping untuk melihat koordinat (bounding box) setiap wajah
        for i, detection in enumerate(detections):
            bbox = detection.bounding_box
            print(f"Wajah {i+1}: Titik X={bbox.origin_x}, Titik Y={bbox.origin_y}, Lebar={bbox.width}, Tinggi={bbox.height}")

            # Ambil nilai pixel secara langsung, tidak perlu dikali W dan H lagi
            x1, y1, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            # 2. GAMBAR KOTAK DI GAMBAR NUMPY ARRAY (img_bgr)
            #cv2.rectangle(img_bgr, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            
            # x1 = int(x1 * W)
            # y1 = int(y1 * H)
            # w = int(w * W)
            # h = int(h * H)

            # #blur face
            # img_bgr[y1:y1+h, x1:x1+w, :] = cv2.blur(img_bgr[y1:y1+h, x1:x1+w, :], (20, 20))
            # Tentukan titik akhir X dan Y (x2 dan y2)
            x2 = x1 + w
            y2 = y1 + h

            # 🚨 PENCEGAHAN ERROR: Pastikan koordinat tidak keluar dari batas gambar!
            # max(0, nilai) memastikan angka terkecil adalah 0 (tidak ada minus)
            # min(Batas, nilai) memastikan angka tidak melebihi resolusi gambar
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2) # W adalah lebar gambar (Width) dari mp_image.width
            y2 = min(H, y2) # H adalah tinggi gambar (Height) dari mp_image.height

            # Validasi terakhir: Lakukan blur hanya jika area masuk akal (lebar dan tinggi > 0)
            if x2 > x1 and y2 > y1:
                # Ambil area wajah (Region of Interest / ROI)
                face_roi = img_bgr[y1:y2, x1:x2]
                
                # Terapkan blur pada area tersebut. 
                # Tips: Angka (50, 50) bisa dibesarkan jika dirasa kurang blur
                blurred_face = cv2.blur(face_roi, (500, 500))
                
                # Masukkan kembali wajah yang sudah diblur ke gambar asli
                img_bgr[y1:y2, x1:x2] = blurred_face
        
            print(f"    Detection #{i}:")

            # --- KODE UNTUK MENCETAK OUTPUT SESUAI DOKUMENTASI ---
            print("FaceDetectionResult:")
            print("  Detections:")

            # 1. Bounding Box (Kotak Wajah)
            print("      BoundingBox:")
            print(f"        origin_x: {bbox.origin_x}")
            print(f"        origin_y: {bbox.origin_y}")
            print(f"        width: {bbox.width}")
            print(f"        height: {bbox.height}")
        
            # 2. Categories (Tingkat Kepercayaan / Score)
            print("      Categories:")
            for j, category in enumerate(detection.categories):
                print(f"        Category #{j}:")
                # Default ke 0 jika index kosong
                idx = category.index if category.index is not None else 0
                print(f"          index: {idx}")
                print(f"          score: {category.score}")
                
            # 3. Normalized Keypoints (Titik Penting Wajah seperti Mata, Hidung, dll)
            print("      NormalizedKeypoints:")
            if detection.keypoints:
                for k, kp in enumerate(detection.keypoints):
                    print(f"        NormalizedKeypoint #{k}:")
                    print(f"          x: {kp.x}")
                    print(f"          y: {kp.y}")

    # 3. TAMPILKAN GAMBAR YANG SUDAH DIGAMBAR (img_bgr)
    # Tambahkan fungsi resize sementara jika gambar terlalu besar untuk layar
    # img_bgr_resized = cv2.resize(img_bgr, (int(W/3), int(H/3))) 
    
    # cv2.imshow("Hasil Deteksi", img_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

cv2.imwrite(os.path.join(output_dir, 'hasil_blur.jpg'), img_bgr)