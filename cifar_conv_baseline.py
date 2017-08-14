''' 
Baseline neural network for classification of CIFAR10 images 

Architecture: 
    Conv -> Pool -> Conv -> Pool -> Conv -> Pool -> Flatten -> FC -> ReLu -> Softmax  
    Loss function = Cross entropy 

Source for n/w : https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
Initial Learning rate = 0.001 (TODO)
'''

import theano 
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.compute_test_value = 'warn'
import theano.tensor as T
import numpy as np
import gzip, cPickle, math
from tensorboard_logging import Logger
from tqdm import *
from time import time 
from layers import Dense, Activation, Dropout, Conv2D, Pool2D, Flatten
from utils import get_cifar10, ModelCheckpoint, load_model, unpickle
import argparse

parser = argparse.ArgumentParser(description="baseline neural network for mnist")
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument("--batch_size", type=int, help="Batchsize value (different for local/prod)")
args = parser.parse_args()

def main():


    train_x, train_y, valid_x, valid_y, test_x, test_y = get_cifar10('./cifar-10-batches-py/')
    labels = unpickle('./cifar-10-batches-py/batches.meta')['label_names']

    num_epochs = args.epochs
    eta        = 0.001
    batch_size = args.batch_size

    # input 
    x = T.tensor4("x")
    y = T.ivector("y")
    x.tag.test_value = np.random.randn(5, 3, 32, 32)
    # network definition 
    conv1 = Conv2D(input=x, num_filters=16, input_channels=3, size=3, strides=(1,1), padding=1,  name="conv1")
    act1  = Activation(input=conv1.output, activation="relu", name="act1")
    pool1 = Pool2D(input=act1.output, stride=(2,2), name="pool1")
    
    conv2 = Conv2D(input=pool1.output, num_filters=20, input_channels=16, size=3, strides=(1,1), padding=1,  name="conv2")
    act2  = Activation(input=conv2.output, activation="relu", name="act2")
    pool2 = Pool2D(input=act2.output, stride=(2,2), name="pool2")

    conv3 = Conv2D(input=pool2.output, num_filters=32, input_channels=20, size=3, strides=(1,1), padding=1,  name="conv3")
    act3  = Activation(input=conv3.output, activation="relu", name="act3")
    pool3 = Pool2D(input=act3.output, stride=(2,2), name="pool3")

    flat  = Flatten(input=pool3.output)
    fc1   = Dense(input=flat.output, n_in=32*4*4, n_out=10, name="fc1")
    softmax  = Activation(input=fc1.output, activation="softmax", name="softmax")





if __name__ == '__main__':
    main()