import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os # supaya python sabi beriinteraksi dengan sistem komputer
import argparse # memungkinkan puthon bisa nerima perintah langsung dari terminalnya

# ngecek ada folter output ga, kalau ga ada, buat folder
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
def process_img(img, detector):
    # 1. Ambil dimensi dari gambar asli OpenCV (Numpy Array)
    H, W = img.shape[:2]

    # 2. Konversi warna BGR (OpenCV) ke RGB (MediaPipe)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Bungkus numpy array tersebut menjadi objek mp.Image 
    # supaya bisa diolah mediapipe
    mp_image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # 4. Lakukan deteksi wajah pada frame webcam saat ini
    face_detector_result = detector.detect(mp_image_frame) # nyuruh ai analisis mp_image_frame
    # hasilnya itu bentuk laporan objek besar gitu
    detections = face_detector_result.detections # nah ini bilang cuma ambil atribut .detections aja
    # karena cuma mau deteksi wajah doang (berarti banyak atribut yang sabi diambil)
    
    if not detections: # kalau ga nemu wajah, dibalikin list kosong
        # Saya hilangkan print di sini agar terminalmu tidak 'banjir' teks saat webcam menyala
        pass # jadi skip perintah blurringny
    else:
        for detection in detections:
            # model blaze face yang dipake otomatis ngasilin koordinatnya
            bbox = detection.bounding_box # ini kita mau akses titik koordinatnya
            
            x1, y1, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            x2 = x1 + w
            y2 = y1 + h

            # PENCEGAHAN ERROR: Pastikan koordinat tidak keluar dari batas gambar
            x1 = max(0, x1) # akan memilih angka yang paling besar di antara A dan B.
            y1 = max(0, y1)
            x2 = min(W, x2) # akan memilih angka yang paling kecil di antara A dan B.
            y2 = min(H, y2) 

            # Validasi area
            if x2 > x1 and y2 > y1:
                # 5. Langsung potong wajah dari 'img' asli (karena img sudah Numpy BGR)
                face_roi = img[y1:y2, x1:x2]
                
                # Terapkan blur. Angka (50, 50) biasanya sudah cukup baik untuk video
                blurred_face = cv2.blur(face_roi, (100, 100))
                
                # Masukkan kembali wajah yang sudah diblur
                img[y1:y2, x1:x2] = blurred_face
                
    # Kembalikan gambar yang sudah diproses
    return img

# ini buat pelayan restoran yang siap nyatet pesanan kita
args = argparse.ArgumentParser()

args.add_argument("--mode", default='webcam')
#args.add_argument("--filePath", default='i./data/person.jpg')
args.add_argument("--filePath", default=None)

args = args.parse_args()


# detect face

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ini buat detektornya
options = FaceDetectorOptions(
    #modelnya yang udah ditrain google
    base_options=BaseOptions(model_asset_path='./data/blaze_face_full_range.tflite'), # ini pake model
    running_mode=VisionRunningMode.IMAGE) # ini jadi bilang yang masuk gambar
    # meskipun yang masuk video, nanti akan diakses per frame

# ini nyalain detektornya
with FaceDetector.create_from_options(options) as detector:
   
    if args.mode in ["image"]:

        img_path = cv2.imread(os.path.join('.','data', 'person.jpg'))


        img = process_img(img_path, detector)


        cv2.imwrite(os.path.join(output_dir, 'hasil_blur.jpg'), img)

    elif args.mode in ["video"]:


        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'hasil_blur.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame.shape[1], frame.shape[0]))

        while ret:
            frame = process_img(frame,detector)

            output_video.write(frame)

            ret, frame = cap.read()

            
        cap.release()
        output_video.release()

    
    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        while ret:

            frame = process_img(frame,detector)

            cv2.imshow('frame', frame)

            #cv2.waitKey(25)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()
        