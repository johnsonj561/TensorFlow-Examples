### CIFAR-10 Conv Net Example

This model was taken from [TensorFlow's CNN Tutorial]

Dataset from [CIFAR-10] contains total of 60K 32 x 32 RGB  images acrossed 10 categories, balanced with 6K per class.

Model is similar to AlexNet, few changes to top layers.

Architecture:
- conv1
- pool1
- norm1
- conv2
- norm2
- pool2
- local3
- local4
- softmax linear

Model achieves 86% accuracy in top 1 precision.


[TensorFlow's CNN Tutorial]:(https://www.tensorflow.org/tutorials/images/deep_cnn)
[CIFAR-10]:(https://www.cs.toronto.edu/~kriz/cifar.html)