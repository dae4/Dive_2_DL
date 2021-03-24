#%%
import tensorflow as tf
from func import *

batch_size, num_steps = 32,5
train_iter, vocab = load_data_time_machine(batch_size,num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2

num_epochs, lr = 500, 1
device_name = try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)

lstm_cell = tf.keras.layers.LSTMCell(num_hiddens, kernel_initializer='glorot_uniform')
lstm_layer = tf.keras.layers.RNN(lstm_cell, time_major=True, return_sequences=True, return_state=True, go_backwards=True)
with strategy.scope():
    model = RNNModel(lstm_layer, vocab_size=len(vocab))
train_rnn(model, train_iter, vocab, num_hiddens, lr, num_epochs, strategy)
# %%
