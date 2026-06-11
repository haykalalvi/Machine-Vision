## Day 11 Key Questions

1. What does mAP@0.5 physically mean for a PCB inspection system?
   Answer: mAP@0.5 measures how well the model detects each defect class across all confidence levels, requiring predicted boxes to overlap ground truth by at least 50%. It averages the area under the Precision-Recall curve for each class, then averages across all classes. For PCB inspection it tells us the overall detection quality — a mAP@0.5 of 0.85 means our detector is reliably finding defects at the right locations.

2. Why is recall more important than precision for defect detection?
   Answer: it is more danger to have missing defect(recall) than false alarm(precision)

   False Negative (low recall) → defective part ships to customer
    Consequence: warranty claim, product recall, safety incident,
                 brand damage, regulatory fine
    Cost: potentially millions of dollars

   False Positive (low precision) → good part gets rejected
    Consequence: wasted part, re-inspection time, 
                 slightly lower throughput
    Cost: cost of one part + 30 seconds of operator time

    The cost ratio between these two errors is often 100:1 or higher. This is why in manufacturing you almost always operate at a low confidence threshold (0.2–0.3) to maximize recall, and accept more false alarms.

3. What does mosaic augmentation do and why does it help detection?
   Answer: it is combines four training images into a single image in random ratios, because it forces the model to detect smaller objects
   and adapt do different scales

4. If your model has mAP@0.5 = 0.82 but recall = 0.71,
   what would you do to improve recall without retraining?
   Answer: adjust the threshold. recall is about catching the defects. so, we need to lower the threshold so it can catches the defect more
   aggressive.

5. Why must you export to ONNX before deployment?
   Answer: because it is the universal format for deployment and runs in cpu
   -. the framwork independence. i trained the model with pytorch, factory pc often uses TensorRT (NVIDIA), OpenVINO(Intel) or C++
   ONNX is the universal intermediate representation
   -. Speed optimization. ONNX Runtime applies graph optimizations automatically
   -. INT8 Quantization path.ONNX Runtime applies graph optimizations automatically hence it gives speedup on CPU

6. What is the difference between mAP@0.5 and mAP@0.5:0.95?
   When would you care more about the stricter metric?
   Answer: the differences is on the metrics. mAP@0.5 calculates the mean average precision for IoU thresholds at least 50%. but, the mAP@0.5:0.95
   calculates the mean average precision for IoU thresholds in the range of 50% to 95% for more precise detection. we often care about it if the goals
   of object detection is to detect precise and small objects.

   When mAP@0.5 is sufficient (most manufacturing QC):

   You need to know the defect exists and roughly where it is
   A human operator will verify the exact location
   Pass/fail decision is binary — precise boundary doesn't matter
   NEU metal surface detection → use mAP@0.5

   When mAP@0.5:0.95 matters:

   Robotic grasping — the robot needs the exact object boundary to plan a grasp
   Measurement systems — you're measuring defect area in mm², not just flagging it
   Surgical robotics — precise tissue boundary detection
   Autonomous vehicles — precise pedestrian/vehicle boundaries for path planning