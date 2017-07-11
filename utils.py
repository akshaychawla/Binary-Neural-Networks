import gzip, cPickle

def get_data():
    f = gzip.open('mnist.pkl.gz','rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x , test_y  = test_set
    return train_x, train_y, valid_x, valid_y, test_x, test_y 