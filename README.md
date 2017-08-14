**Note: This is still an ongoing project, changes and results will keep appearing as experiments are run**
 
# Binary Neural Networks
An attempt to recreate neural networks where the weights and activations are binary variables. This is a common underlying theme of the papers [BinaryConnect](https://arxiv.org/abs/1511.00363), [Binarized neural networks](https://arxiv.org/abs/1602.02830) and [XNOR-Net](https://arxiv.org/abs/1603.05279). The goal is to free these high performing deep learning models from the shackles of a supercomputer (read: GPUs) and bring them to edge devices which typically have a much lower memory footprint and limited computation capabilities. 

![Quantize](readme_data/quantize.jpg "Quantize")

### Requirements 
1. Theano (0.9.0 or higher)
2. Python 2.7
3. numpy 
4. tqdm (awesome progbars)
5. tensorboard (logging)

### Idea
1. Regularization: Binarization of weights is a form of noise for the system. Hence just like dropout, binarization can act as a regularizer. 
2. Discretization is a form of corruption, we can make discretization errors in different weights but the randomness of this corruption cancels out these errors. 
3. Perform forward and backward pass using binarized form of weights, however keep a second set of weights (using full precision fp32) for gradient update. This is because SGD makes infinitesimal changes that would be lost due to binarization. For forward pass, sample a set of binary weights from full precision weights using deterministic or stochastic binarization. 
4. Deterministic binarization: W<sub>b</sub> = +1 if W >=0 ; else -1. 
5. Stochastic binarization: (TODO)
6. Clip weights if they exceed +1/-1 as they do not contribute to the network due to the binarization.  


### Experiment 1: MNIST
In this experiment a baseline and binarized version of a standard MLP is trained on the MNIST dataset. In order to have a fair comparison between the two networks, the data is not augmented and dropout is applied to the baseline network to give it a fair chance against the binary network (as the binarization acts as a regularizer). The learning rate is set at a constant 0.001 and both networks are trained for 200 epochs with a batch size of 256. The training loss and validation accuracy is visualized in tensorboard using [this gist](https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514). The basic architecture is as follows:
 
![architecture](readme_data/network_arch.jpg "architecture")

#### Results
**1. Baseline**

<img src="readme_data/baseline_loss.png" alt="Drawing" style="width: 200px;"/>
<img src="readme_data/baseline_acc.png" alt="Drawing" style="width: 200px;"/>
<!-- [Training Loss](readme_data/baseline_loss.png "Training Loss") ![Validation Accuracy](readme_data/baseline_acc.png "Validation Accuracy") !-->
Test accuracy: 0.9687   


**2. Binarized** 

<img src="readme_data/binary_loss.png" alt="Drawing" style="width: 200px;"/>
<img src="readme_data/binary_acc.png" alt="Drawing" style="width: 200px;"/>
<!-- ![Training Loss](readme_data/binary_loss.png "Training Loss") ![Validation Accuracy](readme_data/binary_acc.png "Validation Accuracy") -->
Test accuracy: 0.9372 

### Experiment 2: CIFAR 10
(TODO)

#### Results
(TODO)

### Why?
This repository is the result of my curiosity to fix computation problems that arise with the deployment of (very big) neural networks. I am fascinated by recent research that has come up with effective ways to compress, prune and/or quantize deep networks so that they run in resource constrained environments like ARM chips. This is a (tiny) step in that direction.    

