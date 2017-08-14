import gzip
import cPickle as pickle
import h5py 
import numpy as np
import os 
import re

# def get_best_model(folder):
#   """ Return the filepath of best model hdf5 (weights) file in provided folder"""
#   assert os.path.isdir(folder), "Must be valid folder!"
#   fnames = os.listdir(folder)
#   fnames = [f for f in fnames if f.endswith(".hdf5")]
#   regex = r"^epoch_[0-9]"
#   for fname in fnames:
#       matchObj = re.search(r"^epoch_[0-9]+", fname)
#       epoch_x  = matchObj.group(0)
#       epoch_x  = epoch_x.lstrip("epoch_")

class ModelCheckpoint:
    def __init__(self, folder):
        self.epoch = 0
        self.best_val_acc = 0.0
        self.folder = folder
        self.best_val_acc_filename = ""
        assert os.path.isdir(self.folder), "the Folder passed does not exist on disk!"

    def check(self, val_acc, params):
        
        if val_acc > self.best_val_acc:
            
            filename = os.path.join(self.folder, "epoch_{}_val_acc_{}.hdf5".format(self.epoch, val_acc))
            print "Val acc improved from ", self.best_val_acc, " to ", val_acc, \
            ". Dumping to ", filename
            save_model(path=filename, params=params)
            self.best_val_acc = val_acc
            self.best_val_acc_filename = filename

        else:

            print "Val acc did not improve."
            
        self.epoch += 1


def save_model(path, params):
    filename = path  
    F = h5py.File(filename, "w")
    weights = F.create_group("weights")
    for param in params:
        weights.create_dataset(param.name, data=param.get_value())
    F.close()

def load_model(path, params):
    filename = path 
    F = h5py.File(filename, "r")
    weights = F["weights"]
    for param in params:
        param.set_value(weights[param.name][:])
    F.close()
    print "..model loaded."

def get_mnist():
    f = gzip.open('mnist.pkl.gz','rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x , test_y  = test_set
    return train_x, train_y, valid_x, valid_y, test_x, test_y 

def unpickle(file):
    """ unpickle given and return as dictionary """
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).astype("float")
        Y = np.array(Y)
    return X, Y

def get_cifar10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

    # split training set into train + validation 
    import random 
    all_indices   = range(50000)
    val_indices   = random.sample(all_indices, 10000)
    train_indices = list(set(all_indices) - set(val_indices))

    train_x = Xtr[train_indices]
    train_y = Ytr[train_indices]

    valid_x = Xtr[val_indices]
    valid_y = Ytr[val_indices]

    return train_x, train_y, valid_x, valid_y, Xte, Yte