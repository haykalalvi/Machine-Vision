
WE TALK ABOUT IMPROVING SPEED

## Notes
1. Quantization in onnx runtime tu maksudnya adalah mengkuantiasasi model onnx ke 8 bit integer linear quantization.
   -. Jadi dia mengubah unit measurement dari resepnya ( asumsikan seperti pabrik makanan dengan berbagai kitchen)
   -. setiap weight di model itu kan number. kalau FP32 itu nyimpen nomor dengan 32 bits (4 bytes) precision yang bisa represent 4 billion values berbeda. Sedangkan, INT8 itu cuma 8 bits yang sabi nampilin 256 values berbeda

2. Dynamic quantization
   -.weights di kuantisasi sekali,offline, dan sebelum diproses
   -. tapi activationsnya di kuantisasi saat berjalan
   -. easisest to apply, tapi speednya moderate

3. Static quantization
   -.weights di kuantisasi sekali,offline, dan sebelum diproses
   -. activationsnya di kuantisasi sekali dan offline juga, tapi nanti hasilnya range yang akan digunakan. jadi harus run real data ke model dan ninjau nilai aktual min/max di setiap layernya (calibration namanya)
   -. jadi faster inferencenya, tapi nilai calibration rangenya ya sebagus calibration datanya aja, kalai data kalibrasi kita ga sebagus real imagenya nanti, ya nanti bsia salah

4. QAT (Quantization aware training)
   -. intinya melatih model untuk latihan dengan kuantisasi yang digunakan (INT8) langsung
   -. jadi dia simulate INT8 rounding error saat trainingnya, jadi nanti pas pake int8 ga ada akurasi loss
5. latency itu delay dari cause and effect



   


# 
2. FP32 Baseline (Stage 1 YOLOv8)
  Mean:      25.01 ms
  Median:    24.85 ms
  Std dev:   0.51 ms
  Min/Max:   24.35 / 26.54 ms
  FPS:       40.0
  Model size: 11.84 MB

  --- Comparison with Day 14 ---
  Day 14 (with cold-start mixed in): 50.40ms avg
  Day 15 (properly warmed up):       25.01ms avg
3. Dynamic Quantization result


    Dynamic INT8 (Stage 1 YOLOv8)

  Mean:      66.99 ms
  Median:    66.78 ms
  Std dev:   1.31 ms
  Min/Max:   65.78 / 75.02 ms
  FPS:       14.9
  Model size: 3.34 MB


    DYNAMIC QUANTIZATION RESULTS

  Latency:  25.01ms -> 66.99ms (0.37x speedup)
  FPS:      40.0 -> 14.9
  Size:     11.84MB -> 3.34MB (3.54x smaller)
4. Static result

    
  Static INT8 (Stage 1 YOLOv8)

  Mean:      15.02 ms
  Median:    14.91 ms
  Std dev:   0.63 ms
  Min/Max:   14.14 / 16.95 ms
  FPS:       66.6
  Model size: 3.37 MB

5. Comparison result
    
    THREE-WAY COMPARISON
    Model                   Latency      FPS       Size   Speedup

    FP32 (baseline)          25.01ms    40.0    11.84MB     1.00x
    Dynamic INT8             66.99ms    14.9     3.34MB     0.37x
    Static INT8              15.02ms    66.6     3.37MB     1.67x

6. accuracy summary
    
ACCURACY VALIDATION SUMMARY


FP32:
  Total detections across 48 images: 34
  Avg confidence (when detected): 0.581

Dynamic INT8:
  Total detections across 48 images: 35
  Avg confidence (when detected): 0.575

Static INT8:
  Total detections across 48 images: 34
  Avg confidence (when detected): 0.577


QUANTIZATION IMPACT (vs FP32)


Dynamic INT8:
  Mean IoU vs FP32 boxes:     0.9273  (GOOD)
  Detection count changes:    47/48 images identical
  Images with MORE detections: 1
  Images with FEWER detections: 0

Static INT8:
  Mean IoU vs FP32 boxes:     0.9005  (GOOD)
  Detection count changes:    46/48 images identical
  Images with MORE detections: 1
  Images with FEWER detections: 1

7. Recommendation summary
Recommended: Static INT8
  FPS: 66.6 (vs FP32: 40.0)
  Accuracy retained: 92.0%


## Questions
1. Why does static quantization need calibration data but dynamic doesn't?
   static quantization need calibration data because they calculate the quantization parameter (weights and activations)
2. What information does calibration actually measure?
   Quantization parameter (activations range for each real representative images)
3. Given your Day 14 finding (Stage 1 = 50ms bottleneck, recall = 0.66), what's the risk of quantization for this specific model?
   The lower recall.


2. What does calibration actually measure, concretely?
   Answer: weights and activations of the model

3. Given Day 14's recall=0.66, what's the risk of quantization for THIS
   model specifically? Did your Script 4 results support or contradict
   that concern?
   Answer: it will lower the recall and yeah it happened. even though the accuracy is retained up to 92.%

4. If Dynamic INT8 showed IoU=0.95 but only 1.05x speedup, while Static
   INT8 showed IoU=0.85 but 1.8x speedup -- which would you choose for
   a QC system, and why?
   Answer: depens on the speed needed on the qc system. but, i would choose the dynamic one. because it retained the IoU better.

5. The model-warmup lesson from Day 14 -- how does Script 1's benchmark
   harness specifically address it?
   Answer: it showed the speed is faster up to half the cold start
