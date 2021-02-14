#%%
import tensorflow as tf
import numpy as np

x=np.arange(4.0)
x = tf.Variable(x)
# %%
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
# %%
x_grad = t.gradient(y, x)
x_grad