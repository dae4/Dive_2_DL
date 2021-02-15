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
# %%
x_grad == 4 * x
# %%
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
# %%
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as `y = tf.reduce_sum(x * x)`
# %%
# Set `persistent=True` to run `t.gradient` more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
# %%
t.gradient(y, x) == 2 * x
# %%
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
# %%
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
# %%
d_grad == d / a
# %%
