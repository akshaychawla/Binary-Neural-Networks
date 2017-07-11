''' 
Binary (+-1) neural network for classification of MNIST digits 

Architecture: 
    Input (784) => hidden-1 (2048) + ReLU => hidden-2 (2048) + ReLU => hidden-3 (2048) + ReLU => output (10) + softmax  
    Loss function = Cross entropy 

Paper: https://arxiv.org/pdf/1602.02830.pdf ( Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to +1 or -1)
Initial Learning rate = 0.001
'''

import theano 
import theano.tensor as T
import numpy as np
import gzip, cPickle, math
from tensorboard_logging import Logger
from tqdm import *
from time import time
from layers import BinaryDense, Activation, clip_weights
from utils import get_data

def main():

    train_x, train_y, valid_x, valid_y, test_x, test_y = get_data()

    num_epochs = 250 
    eta        = 0.001
    batch_size = 256

    # input 
    x = T.matrix("x")
    y = T.ivector("y")

    hidden_1 = BinaryDense(input=x, n_in=784, n_out=2048, name="hidden_1")
    act_1    = Activation(input=hidden_1.output, activation="relu", name="act_1")
    hidden_2 = BinaryDense(input=act_1.output, n_in=2048, n_out=2048, name="hidden_2")
    act_2    = Activation(input=hidden_2.output, activation="relu", name="act_2")
    hidden_3 = BinaryDense(input=act_2.output, n_in=2048, n_out=2048, name="hidden_3")
    act_3    = Activation(input=hidden_3.output, activation="relu", name="act_3")
    output   = BinaryDense(input=act_3.output, n_in=2048, n_out=10, name="output")
    softmax  = Activation(input=output.output, activation="softmax", name="softmax")

    # loss
    xent     = T.nnet.nnet.categorical_crossentropy(softmax.output, y)
    cost     = xent.mean()/batch_size # scaling the mean

    # errors 
    y_pred   = T.argmax(softmax.output, axis=1)
    errors   = T.mean(T.neq(y, y_pred))

    # updates + clipping (+-1)
    params_bin   = hidden_1.params_bin + hidden_2.params_bin + hidden_3.params_bin 
    params       = hidden_1.params + hidden_2.params + hidden_3.params
    grads    = [T.grad(cost, param) for param in params_bin] # calculate grad w.r.t binary parameters
    updates  = []
    for p,g in zip(params, grads): # gradient update on full precision weights (NOT binarized wts)
        updates.append(
                (p, clip_weights(p - eta*g)) #sgd + clipping update
            )

    # compiling train, predict and test fxns 
    train   = theano.function(
                inputs  = [x,y],
                outputs = cost,
                updates = updates
            )
    predict = theano.function(
                inputs  = [x],
                outputs = y_pred
            )
    test    = theano.function(
                inputs  = [x,y],
                outputs = errors
            )

    # train 
    logger = Logger("logs/{}".format(time()))
    for epoch in range(num_epochs):
        
        print "Epoch: ", epoch
        
        epoch_hist = {"loss": []}
        
        t = tqdm(range(0, len(train_x), batch_size))
        for lower in t:
            upper = min(len(train_x), lower + batch_size)
            loss  = train(train_x[lower:upper], train_y[lower:upper].astype(np.int32))    
            t.set_postfix(loss="{:.2f}".format(float(loss)))
            epoch_hist["loss"].append(loss.astype(np.float32))
        
        # epoch loss
        average_loss = sum(epoch_hist["loss"])/len(epoch_hist["loss"])         
        t.set_postfix(loss="{:.2f}".format(float(average_loss)))
        logger.log_scalar(
                tag="Training Loss", 
                value= average_loss,
                step=epoch
                )

        # validation accuracy 
        val_acc  =  1.0 - test(valid_x, valid_y.astype(np.int32))
        logger.log_scalar(
                tag="Validation Accuracy", 
                value= val_acc,
                step=epoch
                )        



if __name__ == '__main__':
    main()
