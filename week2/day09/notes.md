## Day 09 Key notes

1. Accuracy is often misleading (especially for QC), because assume this conditions below
    Dataset: 1000 parts total
    Good parts:      950   (95%)
    Defective parts: 50    (5%)

    'Always predict good' model:
    Accuracy: 95.00%  ← looks great!
    Defects caught: 0 out of 50  ← actually useless ( no defect caught)

2. Optimizer is the algorithm that optimize/change ur parameters (weightandbiases)

3. loss function is the grader, it will give you penalty based on the guesses

4. what does pos_weight does in BCEWithLogitsLoss?
    -> it is for adjust the dataset imbalance, 

5. Focal loss use a addition of multiplier to down weighting the easy examples and focus on training the hard examples
   (the solution of no.1 problem) for the easy example, it will contribute small to the loss (down weighting the easy examples).


6. Guessing -> Reality Check -> Blaming(bias, weight, kernel,norm param) -> Adjusting

7. weight decay is to prevent overfitting

## Day 9 Key Questions — answer without looking

1. Why is accuracy a bad metric for manufacturing QC with 5% defect rate?
   Answer: Because of data imbalance, a 'lazy' model can just predict 'Good' for every single part. It will achieve 95% accuracy while catching zero defects, making the metric completely useless for quality control."

2. What does Recall = 0.90 mean in a factory context?
   Answer: the model can correctly identify bad products. Out of every 100 actual defective products on the line, the model successfully catches 90 of them, but lets 10 slip through to the customer.

3. If you lower the decision threshold from 0.5 to 0.2:
   - What happens to recall?
   - What happens to precision?
   - Which error (FN or FP) becomes more common?
   Answer: If you lower the threshold to 0.2, the model only needs to be 20% suspicious to sound the defect alarm. Because it is highly sensitive, it will sound the alarm constantly. Recall goes up, and precision goes down (overly sensitive) -> False positive increase

4. What does AUROC = 0.95 mean physically?
   Answer: If you randomly select one defective part and one good part from the factory line, there is a 95% chance the model will assign a higher 'defect score' to the truly defective part.

5. What does the OneCycleLR scheduler do and why does it help?
   Answer: it will adjust the learning rate in real time on learning progress, It ramps the learning rate up to jump out of local minima, and then scales it smoothly down to settle into the final optimal weights.

6. What is gradient clipping and when do you need it?
   Answer: it is a mathematical safety net for backpropagation process, it prevents the optimizer to take big steps

