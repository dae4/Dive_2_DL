import torch
from torch import nn
from torch.nn import functional as F
import math

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)