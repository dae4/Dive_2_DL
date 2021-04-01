#%%
import tensorflow as tf

def comp_conv2d(conv2d, X):
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    return tf.reshape(Y, Y.shape[1:3])


conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape ## if padding is same, not change shape 
# %%
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='valid')
comp_conv2d(conv2d, X).shape ## padding is None 
# %%

conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape ## shape / strides ==> (8,8) / 2 = (4,4)
# %%
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
# %%
