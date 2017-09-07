''' 
Baseline neural network for classification of CIFAR10 images 

Architecture: 
    Conv -> Pool -> Conv -> Pool -> Conv -> Pool -> Flatten -> FC -> ReLu -> FC -> Softmax  
    Loss function = Cross entropy 

Source for n/w : https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
'''

import theano 
# theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
# theano.config.compute_test_value = 'warn'
import theano.tensor as T
import numpy as np
import gzip, cPickle, math
from tensorboard_logging import Logger
from tqdm import *
from time import time 
from layers import Dense, Activation, Dropout, Conv2D, Pool2D, Flatten
from utils import get_cifar10, ModelCheckpoint, load_model, unpickle
import ipdb
from sklearn import preprocessing
import argparse

parser = argparse.ArgumentParser(description="baseline neural network for mnist")
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument("--batch_size", type=int, help="Batchsize value (different for local/prod)")
parser.add_argument("--lr", type=float, help="Initial learning rate")
args = parser.parse_args()

def main():


    train_x, train_y, valid_x, valid_y, test_x, test_y = get_cifar10('./cifar-10-batches-py/')
    labels = unpickle('./cifar-10-batches-py/batches.meta')['label_names']
    
    train_x = train_x.astype(np.float32) / 255.0
    valid_x = valid_x.astype(np.float32) / 255.0
    test_x  = test_x.astype(np.float32) / 255.0
    
    
    num_epochs = args.epochs
    eta        = args.lr
    batch_size = args.batch_size

    # input 
    x = T.tensor4("x")
    y = T.ivector("y")
    drop_switch = T.scalar("drop_switch")

    # test values
    # x.tag.test_value = np.random.randn(6, 3, 32, 32).astype(np.float32)
    # y.tag.test_value = np.array([1,2,1,4,5]).astype(np.int32)
    # x.tag.test_value = x.tag.test_value / x.tag.test_value.max()

    # drop_switch.tag.test_value = 1.0
    # import ipdb; ipdb.set_trace()

    # network definition 
    conv1 = Conv2D(input=x, num_filters=50, input_channels=3, size=3, strides=(1,1), padding=1,  name="conv1")
    act1  = Activation(input=conv1.output, activation="relu", name="act1")
    pool1 = Pool2D(input=act1.output, stride=(2,2), name="pool1")
    
    conv2 = Conv2D(input=pool1.output, num_filters=100, input_channels=50, size=3, strides=(1,1), padding=1,  name="conv2")
    act2  = Activation(input=conv2.output, activation="relu", name="act2")
    pool2 = Pool2D(input=act2.output, stride=(2,2), name="pool2")

    conv3 = Conv2D(input=pool2.output, num_filters=200, input_channels=100, size=3, strides=(1,1), padding=1,  name="conv3")
    act3  = Activation(input=conv3.output, activation="relu", name="act3")
    pool3 = Pool2D(input=act3.output, stride=(2,2), name="pool3")

    flat  = Flatten(input=pool3.output)
    drop4 = Dropout(input=flat.output, p=0.5, drop_switch=drop_switch)
    fc1   = Dense(input=drop4.output, n_in=200*4*4, n_out=500, name="fc1")
    act4  = Activation(input=fc1.output, activation="relu", name="act4")
    drop5 = Dropout(input=act4.output, p=0.5, drop_switch=drop_switch)
    fc2   = Dense(input=drop5.output, n_in=500, n_out=10, name="fc2")
    softmax  = Activation(input=fc2.output, activation="softmax", name="softmax")

    # loss
    xent     = T.nnet.nnet.categorical_crossentropy(softmax.output, y)
    cost     = xent.mean()
    
    # errors 
    y_pred   = T.argmax(softmax.output, axis=1)
    errors   = T.mean(T.neq(y, y_pred))

    # updates 
    params   = conv1.params + conv2.params + conv3.params + fc1.params 
    grads    = [T.grad(cost, param) for param in params]
    updates  = []
    for p,g in zip(params, grads):
        updates.append(
                (p, p - eta*g) #sgd
            )

    # compiling train, predict and test fxns     
    train   = theano.function(
                inputs  = [x,y,drop_switch],
                outputs = cost,
                updates = updates
            )
    predict = theano.function(
                inputs  = [x,drop_switch],
                outputs = y_pred
            )
    test    = theano.function(
                inputs  = [x,y,drop_switch],
                outputs = errors
            )

    # train 
    checkpoint = ModelCheckpoint(folder="snapshots")
    logger = Logger("logs/{}".format(time()))
    for epoch in range(num_epochs):
        
        print "Epoch: ", epoch
        print "LR: ", eta
        epoch_hist = {"loss": []}
        
        t = tqdm(range(0, len(train_x), batch_size))
        for lower in t:
            upper = min(len(train_x), lower + batch_size)
            loss  = train(train_x[lower:upper], train_y[lower:upper].astype(np.int32), 1.0) # drop     
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
        val_acc  =  1.0 - test(valid_x, valid_y.astype(np.int32), 0.0) # nodrop
        print "Validation Accuracy: ", val_acc
        logger.log_scalar(
                tag="Validation Accuracy", 
                value= val_acc,
                step=epoch
                )  
        checkpoint.check(val_acc, params)

    # Report Results on test set (w/ best val acc file)
    best_val_acc_filename = checkpoint.best_val_acc_filename
    print "Using ", best_val_acc_filename, " to calculate best test acc."
    load_model(path=best_val_acc_filename, params=params)
    test_acc = 1.0 - test(test_x, test_y.astype(np.int32), 0.0) # no drop 
    print "Test accuracy: ",test_acc

    
if __name__ == '__main__':
    main()
