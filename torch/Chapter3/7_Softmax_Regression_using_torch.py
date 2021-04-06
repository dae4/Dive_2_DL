#%%
import torch
from torch import nn
from func import *
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

## Weight initialization

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

loss = nn.CrossEntropyLoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
train(net, train_iter, test_iter, loss, num_epochs, trainer)
# %%
