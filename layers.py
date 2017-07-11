import theano 
import theano.tensor as T
import numpy as np
import gzip, cPickle, math
from tensorboard_logging import Logger
from tqdm import *
from time import time 

# layer utilities
def binarize(W):
    ''' Convert the float32 data to +-1 '''
    Wb = T.cast(T.where(W>=0.0, 1, -1), dtype=theano.config.floatX)
    return Wb

def clip_weights(w):
    return T.clip(w, -1.0, 1.0)

# Deep learning layers
class Dense():
    def __init__(self, input, n_in, n_out, name):
        self.W = theano.shared(
                np.random.randn(n_in, n_out),
                name  = "W_" + name
                )
        self.b = theano.shared(
                np.zeros((n_out,)),
                name  = "b_" + name, 
                )

        self.input  = input 
        self.params = [self.W, self.b]
        self.output = T.dot(self.input, self.W) + self.b

class Activation():
    def __init__(self, input, activation, name):
        self.input = input
        
        if activation   == "relu":
            self.output = T.nnet.relu(self.input)
        elif activation == "softmax":
            self.output = T.nnet.softmax(self.input)

class BinaryDense():
    def __init__(self, input, n_in, n_out, name):
        self.W = theano.shared(
                np.random.randn(n_in, n_out),
                name  = "W_" + name
                )
        self.b = theano.shared(
                np.zeros((n_out,)),
                name  = "b_" + name, 
                )
        self.Wb         = binarize(self.W)
        self.input      = input 
        self.params     = [self.W, self.b]
        self.params_bin = [self.Wb, self.b]
        self.output     = T.dot(self.input, self.Wb) + self.b

