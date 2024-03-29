#%%
from Chapter4.func import *
import tensorflow as tf

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
# %%
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.01))

params = [W1, b1, W2, b2]
# %%
def relu(X):
    return tf.math.maximum(X, 0)
# %%
def net(X):
    X = tf.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2
# %%
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)
# %%
num_epochs, lr = 10, 0.1
updater = Updater([W1, W2, b1, b2], lr)
train(net, train_iter, test_iter, loss, num_epochs, updater)
# %%
predict(net, test_iter)
# %%
