import numpy as np
import matplotlib.pyplot as plt 
import argparse
import csv, os, sys

parser = argparse.ArgumentParser(description="plot the validation accuracy and training loss history")
parser.add_argument("--loss", type=str, help="location of .csv of training loss")
parser.add_argument("--val_acc", type=str, help="location of .csv of validation accuracy")
args = parser.parse_args()

def main():
    
    # sanity check
    assert os.path.isfile(args.loss), "provided loss file is incorrect"
    assert os.path.isfile(args.val_acc), "provided val_acc file is incorect"

    # load data
    with open(args.loss,"r") as f:
        reader = csv.reader(f)
        loss = []
        for row in reader:
            loss.append(row)
        loss = loss[1:]
    
    with open(args.val_acc,"r") as f:
        reader = csv.reader(f)
        acc    = []
        for row in reader:
            acc.append(row)
        acc = acc[1:]

    assert len(loss) == len(acc), "Loss and validation accuracy containers have different lengths"
    
    # compare number of epochs  
    extract_epochs = lambda x: x[1]
    acc_epochs     = map(extract_epochs, acc)
    loss_epochs    = map(extract_epochs, loss)
    assert cmp(acc_epochs, loss_epochs) == 0, "acc and loss do not have the same epochs"
    
    loss = map(lambda x: float(x[2]), loss)
    acc  = map(lambda x: float(x[2]), acc )
    
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(range(len(loss)), loss)
    plt.xlabel("epoch")
    plt.title("loss")
    plt.subplot(122)
    plt.plot(range(len(acc)), acc)
    plt.xlabel("epoch")
    plt.title("validation accuracy")
    plt.show()
    
if __name__ == "__main__":
    main()
