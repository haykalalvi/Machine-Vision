
there are three common method of transfer learning
1. Feature extraction (only train the last layer (FC layer))
2. partial fine tuning 
3. full fine tuning

=======================================================
RESULTS SUMMARY
=======================================================
  Feature Extraction     → Acc: 0.986 | F1: 0.986 | Time: 5.8min
  Partial Fine-tune      → Acc: 0.986 | F1: 0.986 | Time: 6.7min
  Full Fine-tune         → Acc: 1.000 | F1: 1.000 | Time: 9.3min

Best strategy -> Full Fine-tune, because we do have a bunch amount of datasets, so it will make the full fine tune possible and give the best results


## Questions

1. Why do we freeze early layers but unfreeze later ones?
    this one is one of the type of transfer learning. we use pretrained model that already use "trusted" and "reliable" datasets
    but, if we have data domain that slightly different and we have meduim data set, we can do this method
2. What does requires_grad = False do physically during backpropagation?
    it will disable the gradien descent that optimize the parameters for each layers. often used for  feature extraction method 
3. Why must you use a smaller learning rate when fine-tuning vs training from scratch?
    because on the fine tuned model, the weights already trained, so we dont want to change it a lot
4. why do we do not need augmentation in val data?
    well, back to the roots of the goals of augmentation, to modify the data hence the model can do generalization better, and generalization
    happens on training. so, it is better to do it only on training data
5. For the NEU dataset tomorrow (Day 11 detection): which strategy will i start with?
    i think i am gonna start with full fine tune, because from today's output, i get best results with the full fine tuned method
