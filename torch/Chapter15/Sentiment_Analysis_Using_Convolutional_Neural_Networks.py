import torch
from torch import nn
from func import *


batch_size = 64
train_iter, test_iter, vocab = load_data_imdb(batch_size)

def corr1d(X, K):
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
