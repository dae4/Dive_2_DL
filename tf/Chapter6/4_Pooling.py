#%%
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y

X = tf.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2)) ## default max
# %%
pool2d(X, (2, 2), 'avg')

#%%|
X = tf.reshape(tf.range(16, dtype=tf.float32), (1, 4, 4, 1))
X
# %%
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
pool2d(X)
#%%
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='same', strides=2)
pool2d(X)
#%%
pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='same', strides=(2, 3))
pool2d(X)
# %%
X = tf.reshape(tf.stack([X, X+1], 0), (1, 2, 4, 4))
print(X.shape,"\n",X[:,0,:,:],X[:,1,:,:])
# %%
pool2d = tf.keras.layers.MaxPool2D(3, padding='same', strides=2)
pool2d(X)
# %%
