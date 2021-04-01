#%%
import tensorflow as tf
from func import *

def corr2d_multi_in(X, K):
    return tf.reduce_sum([corr2d(x, k) for x, k in zip(X, K)], axis=0)


X = tf.constant([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = tf.constant([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
#%%

def corr2d_multi_in_out(X, K):
    return tf.stack([corr2d_multi_in(X, k) for k in K], 0)


K = tf.stack((K, K + 1, K + 2), 0)
K.shape
#%%
corr2d_multi_in_out(X, K)

#%%
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = tf.reshape(X, (c_i, h * w))
    K = tf.reshape(K, (c_o, c_i))
    Y = tf.matmul(K, X)  # Matrix multiplication in the fully-connected layer
    return tf.reshape(Y, (c_o, h, w))

X = tf.random.normal((3, 3, 3), 0, 1)
K = tf.random.normal((2, 3, 1, 1), 0, 1)
#%%
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(tf.reduce_sum(tf.abs(Y1 - Y2))) < 1e-6
# %%
