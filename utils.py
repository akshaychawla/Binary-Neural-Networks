import gzip, cPickle
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

def get_data():
    f = gzip.open('mnist.pkl.gz','rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x , test_y  = test_set
    return train_x, train_y, valid_x, valid_y, test_x, test_y 