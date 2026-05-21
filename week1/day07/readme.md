# Day 7: Classical Defect Detector

## Problem
Detect surface defects on hazelnut using classical computer vision only.
Dataset: MVTec AD — industry standard anomaly detection benchmark.

## Approach
Reference-based anomaly detection:
1. Build a per-pixel mean and std image from all good training samples
2. For each test image, compute a z-score deviation map (substract test image with the avg image then divides it with the deviation image)
3. The 95th percentile of the deviation map = the anomaly score
4. Threshold the score to get pass/fail

## Results
| Threshold | AUROC | Precision | Recall | F1 |
|-----------|-------|-----------|--------|----|
| 1.5       | 0.1107|   0.36    |  0.47  | 0.41       
| 1.8       | 0.1107|   0.31    |  0.35  | 0.33       
| 2.0       | 0.1107|   0.25    |  0.24  | 0.24       
| 2.5       | 0.1107|   0.39    |  0.35  | 0.25        
| 3.0       | 0.1107|   0.13    |  0.36  | 0.19   
Higher is better for precision and recall
AUROC closes to 1 better

Precision for detect false positive
Recall for detect false negative
F1 is a harmonic mean (a balance) between Precision and Recall. It is often used instead of Accuracy when  have imbalanced data.

## What I Learned
- Why AUROC is a better metric than accuracy for imbalanced QC datasets
    - It measures how good your system is at separating the two groups
    - AUROC doesn't care about your threshold. It evaluates the underlying math (the anomaly scores themselves). It tells you, "Regardless of where you   draw the line, how well did the scores separate the good from the bad?" It grades the system's potential.
- How natural variation in good samples (std map) helps reduce false positives
- Why classical methods struggle when defects look similar to texture variation
- What would need to change to make this work in real-time (FPS analysis)

## Limitations & Next Steps
- This method fails when defects have the same texture as normal material
- No spatial alignment — if the part is rotated, scores will be wrong
- Week 2 will address this with deep learning anomaly detection (PatchCore)