## Notes
1. === Stage 1 Results ===
    mAP50:    0.6084
    mAP50-95: 0.3353
    Recall:   0.6595 
2. Crop padding are used to expand the bounding box that created by yolo a little bit larger. it will give the unet more context and prevent
   the yolo model to crop defect (sometimes yolo boxes slightly too tight). 
3. This is the pipeline test result 
   === Pipeline test on 002.png ===
    Detections: 1
    cut        conf=0.370 bbox=[154, 294, 195, 427] defect_area=9.7%
    Timing: detect=65.4ms, segment=342.1ms, total=409.0ms
    FPS estimate: 2.4
4. Full pipeline benchmark
    Benchmarking on 60 images...

    Average timing over 60 images:
    Stage 1 (detect):  50.40 ms
    Stage 2 (segment): 19.27 ms  (only when detections found)
    Total:             69.84 ms
    Estimated FPS:     14.3 (pass)

    Average detections per image: 0.82

    Saved: output/03_benchmark.jpg

    PASS: 14.3 FPS vs 10 FPS target

5. Notice from the point 1 and 4 that are a big differences about the time consumed. it is because the warm up inference on startup (point 1)


## Day 14 Key Questions

1. Why does Stage 2's accuracy depend on Stage 1's RECALL specifically
   (not precision)?
   Answer: because if the stage 1 not detect it (low recall), the specific defect wont be fed to the stage 2

2. What would happen to the pipeline if crop_padding=0 and a defect
   sits exactly on the box edge?
   Answer: the defect would be cropped. and the segmentation part (stage 2) wouldnt be able to mask the defect part fully because the information
   is destroyed

3. Looking at your benchmark: which stage is the bottleneck? If you
   needed to hit 30 FPS, what would you optimize first?
   Answer: i would optimize the yolo model. because it took more time compare to the segmentation (but we must carefully not lowering the recall)

4. This pipeline can't detect defect types outside Stage 1's training
   classes. How does Day 13's PatchCore address this exact gap?
   Answer: the patchcore only learn from the good images. so it will compare all the defect type that possible to the good images.
   not based on the supervised or label from the stage 1.

5. In your model card, you set a Stage 1 confidence threshold. Using
   your Day 11 threshold-analysis knowledge, justify your choice.
   Answer: the lower the threshold, it will be more sensitive. it will increase the recall and lower the precision. because i want to make sure
   no defect are being sent or passed.