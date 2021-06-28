#%%
import torch
from torch import nn
from func import *

batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

net = BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                    num_layers=2, dropout=0.2, key_size=128, query_size=128,
                    value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128)
devices = try_all_gpus()
loss = nn.CrossEntropyLoss()
