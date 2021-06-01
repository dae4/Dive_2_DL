#%%
import torch
from torch import nn
from func import *

def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

X = torch.tensor([[0., 1], [2, 3]])
K = torch.tensor([[0., 1], [2, 3]])
trans_conv(X, K)
# %%
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
# %%
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
tconv(X)
# %%
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
tconv(X)
# %%
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
# %%
X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[0, 1], [2, 3]])
Y = corr2d(X, K)
Y
#%%
def kernel2matrix(K):
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
# %%
Y == torch.mv(W, X.reshape(-1)).reshape(2, 2)
# %%
X = torch.tensor([[0.0, 1], [2, 3]])
Y = trans_conv(X, K)
Y == torch.mv(W.T, X.reshape(-1)).reshape(3, 3)
# %%
