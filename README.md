# Dynamic-Adjustment-of-the-Pruning-Threshold-in-Deep-Compression


A PyTorch implementation of the paper "Dynamic-Adjustment-of-the-Pruning-Threshold-in-Deep-Compression".  
This code is built on [Deep-Compression-Pytorch](https://github.com/mightydeveloper/Deep-Compression-PyTorch.git). So if you need requirements for this project, check here.

### Abstract
Recently, convolutional neural networks (CNNs) have been widely utilized due to their outstanding performance in various computer vision fields. However, due to their computational-intensive and high memory requirements, it is difficult to deploy CNNs on hardware platforms that have limited resources, such as mobile devices and IoT devices. To address these limitations, neural network compression research is underway to reduce the size of neural networks while maintaining their performance. This paper proposes a CNN compression technique that dynamically adjusts the thresholds of pruning, one of the neural network compression techniques. Unlike conventional pruning that experimentally or heuristically sets the thresholds that determine the weights to be pruned, the proposed technique can dynamically find the optimal thresholds that prevent accuracy degradation and output the lightweight neural network in less time. To validate the performance of the proposed technique, the LeNet was trained using the MNIST dataset and the light-weight LeNet could be automatically obtained 1.3 to 3 times faster without loss of accuracy.  
![sensitivity](https://github.com/vennie2lee/Dynamic-Adjustment-of-the-Pruning-Threshold-in-Deep-Compression/assets/139102697/4b016d0c-8b1c-492a-893a-2b841794b88e)


### Dataset
I use CIFAR-10 dataset as a training set and test on MNIST dataset.

### Usage
Run
```
python VC_pruning.py
```

### Note
DC_pruning and VC_pruning mean 'DenseNet with CIFAR-10', 'VGG16 with CIFAR-10'

