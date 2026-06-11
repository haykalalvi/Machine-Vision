
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