import theano 
from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse
import theano.tensor as T
import theano.tensor.signal.pool as pool # This is required or else signal module is unavailable
import numpy as np
import gzip, cPickle, math
from tensorboard_logging import Logger
from tqdm import *
from time import time 
import ipdb

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
        
        filter_shape = (n_in, n_out)
        self.W = theano.shared(
                Dense.he_normal(filter_shape),
                name  = "W_" + name
                )
        self.b = theano.shared(
                np.zeros((n_out,)),
                name  = "b_" + name, 
                )

        self.input  = input 
        self.params = [self.W, self.b]
        self.output = T.dot(self.input, self.W) + self.b
    
    @staticmethod
    def he_normal(filter_shape):
        """ Return weights having mean = 0.0 and a limit (variance) = sqrt(2/fan_in) """
        fan_in, fan_out = filter_shape
        fan_in, fan_out = float(fan_in), float(fan_out)
        weights = np.random.normal(loc = 0.0, scale = np.sqrt(2/fan_in), size = filter_shape)
        return weights


class Conv2D():
    def __init__(self, input, num_filters, input_channels, size, strides, padding, name):
        
        filter_shape = (num_filters, input_channels, size, size)
        self.W = theano.shared(
                Conv2D.he_normal(filter_shape),
                name = "W_" + name  
                )
        self.b = theano.shared(
                np.zeros((filter_shape[0],)),
                name = "b_" + name
                )
        self.input = input 
        self.params = [self.W, self.b]

        self.output = T.nnet.conv2d(self.input, self.W, border_mode=padding, subsample=strides) + self.b.dimshuffle('x', 0, 'x', 'x')
   
    @staticmethod
    def he_normal(filter_shape):
        """ Return weights having mean = 0.0 and a limit (variance) = sqrt(2/fan_in) """
        
        fan_in = np.prod(filter_shape[1:]).astype("float32")
        weights = np.random.normal(loc = 0.0, scale = np.sqrt(2/fan_in), size = filter_shape)
        
        return weights
        

class BinaryConv2D():
    def __init__(self, input, num_filters, input_channels, size, strides, padding, name):
        
        filter_shape = (num_filters, input_channels, size, size)
        self.W = theano.shared(
                Conv2D.he_normal(filter_shape),
                name = "W_" + name  
                )
        self.b = theano.shared(
                np.zeros((filter_shape[0],)),
                name = "b_" + name
                )
        
        self.input = input 
        self.Wb    = binarize(self.W)
        self.params = [self.W, self.b]
        self.params_bin = [self.Wb, self.b]
        self.output = T.nnet.conv2d(self.input, self.Wb, border_mode=padding, subsample=strides) + self.b.dimshuffle('x', 0, 'x', 'x')
   

class Pool2D():
    def __init__(self, input, stride, name):
        self.input = input 
        self.output = pool.pool_2d(input = self.input, ws = stride, ignore_border=True)

class Flatten():
    def __init__(self, input):
        self.input = input 
        self.output = T.flatten(self.input, outdim=2)


class Activation():
    def __init__(self, input, activation, name):
        self.input = input
        
        if activation   == "relu":
            self.output = T.nnet.relu(self.input)
        elif activation == "softmax":
            self.output = T.nnet.softmax(self.input)

class BinaryDense():
    def __init__(self, input, n_in, n_out, name):
        
        filter_shape = (n_in, n_out)
        self.W = theano.shared(
                Dense.he_normal(filter_shape),
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

class Dropout():
    def __init__(self, input, p, drop_switch):
        self.input  = input 
        self.srng   = RandomStreams(seed=234)
        self.rv_n   = self.srng.normal(self.input.shape)
        self.mask   = T.cast(self.rv_n < p, dtype=theano.config.floatX) / p # first  dropout mask, scaled with /p so we do not have to perform test time scaling (source: cs231n) 
        self.output = ifelse(drop_switch>0.5, self.input * self.mask, self.input) # only drop if drop == 1.0

