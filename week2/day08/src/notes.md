## Day 8 Key Questions

1. What is a feature map? 
   Answer: Feature map is an 2d grid of numbers that represents the interesting part/specific patterns that the filter/kernel search

2. Why does MaxPooling help with translation invariance?
   Answer: because it is down sampling the spatial resolution of image that are being processed with keeping the maximum value in the grid
   SO,  It makes the network care about whether a feature exists in a general area, rather than its exact pixel coordinates.
   (translation invariance = the network can recognize a feature even it is shift slightly)

3. If I have a 224×224 image and apply a Conv layer with:
   - 64 filters, kernel 3×3, padding=1, stride=1
   What is the output shape?
   Answer: [batch, 64, 224, 224] because output = ((input-kernel+2xpadding)/stride)+1

4. What does BatchNorm do and why does it help training?
   Answer: it will normalize the activation value, it will prevents gradients from vanishing or exploding during training.
   (mean of 0, standard deviation of 1) .it will make the network learn faster

5. Why is ReLU preferred over sigmoid in hidden layers?
   Answer: it is computationally free and avoide the bottleneck that sigmoid function has
   (Sigmoid function problem) the Sigmoid function squishes all inputs into a range between 0 and 1. If you feed it a very high or very low number, the output curve goes completely flat. When the curve is flat, the gradient is zero, meaning the network completely stops learning (this is the "vanishing gradient" problem). ReLU avoids this by keeping a straight, continuous upward line for all positive numbers.