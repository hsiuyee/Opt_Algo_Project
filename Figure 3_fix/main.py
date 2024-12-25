import torch
from data import getDataLoader
from train import setAllSeed, trainModule


if __name__ == '__main__':

    ##############################################
    # set device and datasets
    ##############################################
    train_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    fileAttributes = [['phishing', 68], ['mushrooms', 112], ['w8a', 123], ['ijcnn1', 22]]
    # fileAttributes = [['phishing', 68]]
    # dsetName = 'ijcnn1'
    # featureSize = 22
    
    ##############################################
    # set the parameters
    ##############################################
    seed = 42
    alpha = 0.9
    regularize = 'l2'
    learning_rate = 0.1
    epochs = 50

    ##############################################
    # load data and train module
    ##############################################
    setAllSeed(seed)
    for fileAttribute in fileAttributes:
        dsetName = fileAttribute[0]
        featureSize = fileAttribute[1]
        tr_loader, te_loader = getDataLoader(dsetName, featureSize)
        trainModule('SGD', dsetName, featureSize, alpha, epochs, learning_rate, 
                    tr_loader, te_loader, regularize, device)
        trainModule('SRG', dsetName, featureSize, alpha, epochs, learning_rate, 
                    tr_loader, te_loader, regularize, device)
    
# run ijcnn1 : python3.12 main.py
# run w8a    : python3.12 main.py --dsetName a9a --featureSize 123
