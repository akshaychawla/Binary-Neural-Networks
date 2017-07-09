''' 
Baseline neural network for classification of MNIST digits 

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

class Affine():
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


def get_data():
    f = gzip.open('mnist.pkl.gz','rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x , test_y  = test_set
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def main():

    train_x, train_y, valid_x, valid_y, test_x, test_y = get_data()

    num_epochs = 10 
    eta        = 0.001
    batch_size = 128

    # input 
    x = T.matrix("x")
    y = T.ivector("y")

    hidden_1 = Affine(input=x, n_in=784, n_out=2048, name="hidden_1")
    act_1    = Activation(input=hidden_1.output, activation="relu", name="act_1")
    hidden_2 = Affine(input=act_1.output, n_in=2048, n_out=2048, name="hidden_2")
    act_2    = Activation(input=hidden_2.output, activation="relu", name="act_2")
    hidden_3 = Affine(input=act_2.output, n_in=2048, n_out=2048, name="hidden_3")
    act_3    = Activation(input=hidden_3.output, activation="relu", name="act_3")
    output   = Affine(input=act_3.output, n_in=2048, n_out=10, name="output")
    softmax  = Activation(input=output.output, activation="softmax", name="softmax")

    # loss
    xent     = T.nnet.nnet.categorical_crossentropy(softmax.output, y)
    cost     = xent.mean()/batch_size # scaling the mean

    # errors 
    y_pred   = T.argmax(softmax.output, axis=1)
    errors   = T.mean(T.neq(y, y_pred))

    # updates 
    params   = hidden_1.params + hidden_2.params + hidden_3.params 
    grads    = [T.grad(cost, param) for param in params]
    updates  = []
    for p,g in zip(params, grads):
        updates.append(
                (p, p - eta*g) #sgd
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
    logger = Logger("logs/")
    for epoch in range(num_epochs):
        
        print "Epoch: ", epoch
        
        epoch_hist = {"loss": []}
        
        t = tqdm(range(0, len(train_x), batch_size))
        for lower in t:
            upper = min(len(train_x), lower + batch_size)
            loss  = train(train_x[lower:upper], train_y[lower:upper].astype(np.int32))    
            t.set_postfix(loss="{:.2f}".format(float(loss)))
            epoch_hist["loss"].append(loss)
        
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
