#%%
import tensorflow as tf
from func import *

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)])

batch_size, num_epochs, lr = 2565, 10, 0.1
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(lr=lr)
train_iter, test_iter = load_data_fashion_mnist(batch_size)
train(net, train_iter, test_iter, loss, num_epochs, trainer)
# %%
