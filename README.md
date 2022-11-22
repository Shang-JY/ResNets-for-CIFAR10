# ResNets-for-CIFAR10

Hello everyone, this is my repo for ResNet MiniProject. In this project, we started with resnet50, modified and tested it gradually, and designed some residual networks with fewer parameters to train on the CIFAR10 dataset. The final model has less than 1M parameters but is still able to achieve over 90% accuracy.

In this repository.

The figure folder stores the images of the training process; 

The data folder stores the CIFAR10 dataset; the history folder stores the records of the training process; 

The snapshot folder stores the snapshots of different epochs of the model and parameters of the network. 

The definition and generation mechanism of the residual network are offloaded in ResNet.py, 

The code for training and testing is in main.py.


The settings and accuracies of each model are as follows. We changed, the type of block, layer depth, number of channels, pooling size, we also want to modify the filter size, skip kernel size.

| Name      | # settings | # params| Test acc |
|-----------|---------:|--------:|:-----------------:|
|ResNet_O   |    ResNet(64, Bottleneck, [3, 4, 6, 3], 4)    | 22M   | 94.76%|
|ResNet_N   |    ResNet(64, BasicBlock, [3, 4, 6], 8)       |  8M   | 94.63%|
|ResNet_B   |    ResNet(64, BasicBlock, [2, 2, 2, 2], 4)    | 11M   | 94.78%|
|ResNet_C   |    ResNet(16, BasicBlock, [2, 2, 2, 2], 4)    | 0.7M  | 91.51%|
|ResNet_P   |    ResNet(16, BasicBlock, [2, 2, 2, 2], 3)    | 0.7M  | 91.18%|

![alt text](https://github.com/Shang-JY/ResNets-for-CIFAR10/blob/main/figure/resnet_P_progress.png)


In addition, we conducted some comparison experiments on the training strategies. The effects of image augmentation, learning rate, batch_size, optimizer, scheduler, regularizer etc. However, time is limited and there are limited tests on this part, so if you are interested, we have reserved the interface for you to experiment.

If you have any questions, please contact me.

Jiayi Shang.

Email: js11640@nyu.edu
