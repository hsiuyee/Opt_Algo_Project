import torch
import numpy as np
from sklearn.datasets import load_svmlight_file
from torch.utils.data import TensorDataset, DataLoader


def worker_init_fun(worker_id=0):
    seed = 42
    np.random.seed(seed + worker_id)
    

def loadData(filePath, featureSize):
    dsX, dsY = load_svmlight_file(filePath,
                                  n_features=featureSize)
    tensorX = torch.Tensor(dsX.toarray())
    tensorY = torch.Tensor(dsY)
    maxY = tensorY.amax()
    tensorY = tensorY.eq(maxY).float().reshape(-1, 1)     # normalization
    return tensorX, tensorY


def getDataLoader(dsetName, featureSize, batchSize=128, validSize=128):
    trainSetPath = 'Datasets//' + dsetName + '.txt'
    testSetPath = 'Datasets//' + dsetName + '_t.txt'
    trainSet = TensorDataset(*loadData(trainSetPath, featureSize))
    testSet = TensorDataset(*loadData(testSetPath, featureSize))
    trainLoader = DataLoader(trainSet, batch_size=batchSize,
                             num_workers=0, worker_init_fn=worker_init_fun)
    validLoader = DataLoader(testSet, batch_size=validSize,
                             num_workers=0, worker_init_fn=worker_init_fun)
    return trainLoader, validLoader