#%%
import tensorflow as tf

def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(X[i: i + h, j: j + w] * K))
    return Y

X = tf.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = tf.constant([[0.0, 1.0], [2.0, 3.0]])
## 1 + 6 + 12 = 19
## 2 + 8 + 15 = 25
## 4 + 12 + 21 = 37
## 5 + 14 + 24 = 43

corr2d(X, K)
# %%
## Convolutional Layers
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
#%%
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
#%%
K = tf.constant([[1.0, -1.0]])
#%%
Y = corr2d(X, K)
Y
# %%
corr2d(tf.transpose(X), K)
# %%
## Learning a Kernel
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

print(X, Y)
print(X.shape, Y.shape)
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))

Y_hat = conv2d(X)
print(Y_hat.shape)
## Y_hat.shape = (6-1+1 , 8-2+1)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        ## calculate loss 
        l = (abs(Y_hat - Y)) ** 2
        # Update the kernel
        update = tf.multiply(3e-2, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'batch {i + 1}, loss {tf.reduce_sum(l):.3f}')
#%%
tf.reshape(conv2d.get_weights()[0], (1, 2))
