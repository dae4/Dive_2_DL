#%%
import torch 
from func import *
from torch import nn

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()
    return mask * X / (1.0 - dropout)

X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
# %%
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = torch.random(scale=0.01, size=(num_inputs, num_hiddens1))
b1 = torch.zeros(num_hiddens1)
W2 = torch.random(scale=0.01, size=(num_hiddens1, num_hiddens2))
b2 = torch.zeros(num_hiddens2)
W3 = torch.random(scale=0.01, size=(num_hiddens2, num_outputs))
b3 = torch.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()

# %%
