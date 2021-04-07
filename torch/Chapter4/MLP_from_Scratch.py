import torch
from func import *
from torch import nn

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784 , 10, 256

W1 = nn.parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)*0.01)
b1 = nn.parameter(torch.zeros(num_hiddens,requires_grad=True))
W2 = nn.parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True)*0.01)
b2 = nn.parameter(torch.zeros(num_outputs,requires_grad=True))

params = [W1,b1,W2,b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = torch.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  
    return (H@W2 + b2)

loss = nn.CrossEntropyLoss()

num_epochs, lr =10,0.1
updater = torch.optim.SGD(params, lr=lr)
train(net, train_iter, test_iter, loss, num_epochs, updater)

predict(net, test_iter)