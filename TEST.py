''' This script is used to test the Conv2D layer in theano ''' 

import theano
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.compute_test_value = 'warn'

import numpy as np 
from layers import Conv2D, Pool2D
import theano.tensor as T

def TEST_conv():

    import ipdb; ipdb.set_trace()
    x = T.tensor4("x")
    x.tag.test_value = np.random.randn(1, 3, 32, 32)

    conv2d = Conv2D(x, 1, 3, 3, (1,1), 1, "conv1")

def TEST_pool():

    import ipdb; ipdb.set_trace()
    x = T.tensor4("x")
    x.tag.test_value = np.random.randn(1, 3, 32, 32)

    pool_output = Pool2D(x, (2,2), "pool1")



if __name__ == '__main__':
    TEST_conv()
