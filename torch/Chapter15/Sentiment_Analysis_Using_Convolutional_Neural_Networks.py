import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = load_data_imdb(batch_size)