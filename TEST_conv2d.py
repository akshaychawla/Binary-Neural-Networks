''' This script is used to test the Conv2D layer in theano ''' 

import theano
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.compute_test_value = 'warn'

import numpy as np 
from layers import Conv2D
import theano.tensor as T

def TEST():

	import ipdb; ipdb.set_trace()
	x = T.tensor4("x")
	x.tag.test_value = np.random.randn(1, 3, 32, 32)
	weights_shape = (1,3,3,3) 
	weights = theano.shared(
			value=np.random.randn(*weights_shape),
			borrow=True
			)

	conv2d = Conv2D(x, weights_shape, (1,1), 1,  "conv1")

if __name__ == '__main__':
	TEST()
