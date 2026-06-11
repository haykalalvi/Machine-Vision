## Notes
1. Patch Core is unsupervised learning. it only learns from the good images.
2. Patch core see image closely with magnifying glass to see each part/patch
3. Extract Feature from good images -> store it on memory bank -> compress all the patches in memory bank to reduces the memory
   (For every new image, patch core extract the patch to search the "nearest neighbor" from the memory bank
   if the distance is high-> anomaly)
4. AUROC is used because it is objectively score the model how good to separating into groups. closes to 1 better
5. Image-level AUROC: 1.0000
   Pixel-level AUROC: 0.9918
6. What coreset did -> pick random patches then select another patch that mathematically furthest from the first one. repeat until kept exactly 10%