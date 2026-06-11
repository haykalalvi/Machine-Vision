
## Notes
1. the defect coverage percentage is only 0.78%. this mean the background:defect pixel ratio ≈ 128:1 (too small)
2. with that pixel class imbalance, with crossentropy, the loss function gonna say no defect. we have to use dice loss
3. Loss -> math error rate that computed. used this number to calculate its gradients and learn.
4. IoU & Dice: These are human-readable accuracy scores. They range from 0.0 (0%) to 1.0 (100%). You want these numbers to go UP towards 1.0. 
   Val Dice will always a little bit higher compare to Val IoU.
5. If Val IoU/Val loss & Tr IoU/Tr loss is having the up pattern and down pattern respectively. it indicates overfitting
   (The model only remember the training scratches)


## Results
1. Training complete in 16.7 minutes
2. Best validation IoU: 0.6357
3. Run summary:
    wandb:      epoch 30
    wandb:  train/iou 0.58414
    wandb: train/loss 0.156
    wandb:   val/dice 0.76841
    wandb:    val/iou 0.62422
    wandb:   val/loss 0.13478
4. 1 |   0.7180 |  0.0167 |    0.6214 |   0.0005 |    0.0010 ← best
     2 |   0.5913 |  0.0521 |    0.8199 |   0.0129 |    0.0253 ← best
     3 |   0.5331 |  0.1700 |    0.5195 |   0.1537 |    0.2613 ← best
     4 |   0.4922 |  0.2541 |    0.5172 |   0.1019 |    0.1839
     5 |   0.4583 |  0.2609 |    0.5583 |   0.1029 |    0.1839
     6 |   0.4143 |  0.3251 |    0.4166 |   0.2877 |    0.4434 ← best
     7 |   0.3490 |  0.4025 |    0.3995 |   0.3900 |    0.5552 ← best
     8 |   0.3165 |  0.4273 |    0.4017 |   0.2825 |    0.4333
     9 |   0.2795 |  0.4338 |    0.3232 |   0.3451 |    0.5099
    10 |   0.2386 |  0.4819 |    0.2807 |   0.4389 |    0.6038 ← best
    11 |   0.2448 |  0.4356 |    0.2041 |   0.5424 |    0.7018 ← best
    12 |   0.2263 |  0.4741 |    0.4665 |   0.0594 |    0.1054
    13 |   0.2191 |  0.5025 |    0.2362 |   0.4192 |    0.5870
    14 |   0.1814 |  0.5440 |    0.1791 |   0.5691 |    0.7239 ← best
    15 |   0.2028 |  0.4959 |    0.1756 |   0.5635 |    0.7192
    16 |   0.1700 |  0.5589 |    0.1950 |   0.4891 |    0.6554
    17 |   0.1927 |  0.5226 |    0.2213 |   0.4565 |    0.6224
    18 |   0.1906 |  0.5134 |    0.1541 |   0.5866 |    0.7382 ← best
    19 |   0.1581 |  0.5918 |    0.1507 |   0.5919 |    0.7429 ← best
    20 |   0.1544 |  0.5916 |    0.1423 |   0.6236 |    0.7671 ← best
    21 |   0.1604 |  0.5745 |    0.1362 |   0.6286 |    0.7711 ← best
    22 |   0.1544 |  0.5912 |    0.1436 |   0.6140 |    0.7598
    23 |   0.1629 |  0.5731 |    0.1394 |   0.6135 |    0.7599
    24 |   0.1549 |  0.5823 |    0.1348 |   0.6271 |    0.7702
    25 |   0.1438 |  0.6072 |    0.1328 |   0.6357 |    0.7764 ← best
    26 |   0.1655 |  0.5649 |    0.1359 |   0.6207 |    0.7652
    27 |   0.1206 |  0.6609 |    0.1331 |   0.6341 |    0.7754
    28 |   0.1311 |  0.6385 |    0.1348 |   0.6263 |    0.7697
    29 |   0.1327 |  0.6342 |    0.1355 |   0.6239 |    0.7681
    30 |   0.1560 |  0.5841 |    0.1348 |   0.6242 |    0.7684
4. FINAL SEGMENTATION METRICS    
    Mean IoU:  0.6907
    Mean Dice: 0.8024
    IoU std:   0.1909
    Interpretation: Good


## Questions
1. What problem do skip connections solve?
    it recovers the spatial resolution because of the downsampling, so the segmentation can gets better accuracy
2. Why is Dice Loss better than CrossEntropy for segmentation?
    In segmentation, class imbalance = pixel imbalance
    Defect pixels are RARE compared to background pixels
    This is why we use Dice Loss instead of CrossEntropy

    Average defect coverage across all samples: 0.78%
    This means background:defect pixel ratio ≈ 128:1
    → CrossEntropy would ignore defect pixels (too few)
    → Dice Loss directly optimizes overlap — correct choice
    
3. Why does MVTec only have "good" images in the training set?
    it is almost impossible to predict defect on leather ?