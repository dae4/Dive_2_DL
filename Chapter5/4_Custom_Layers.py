#%%
import tensorflow as tf 

##Layers without Parameters
class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)

layer = CenteredLayer()
layer(tf.constant([1, 2, 3, 4, 5]))

# %%
net = tf.keras.Sequential([
        tf.keras.layers.Dense(128),
        CenteredLayer()
        ])
# %%
Y = net(tf.random.uniform((4,8)))
tf.reduce_mean(Y)
# %%

## Layers with Parameters

class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)


dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
# %%
dense(tf.random.uniform((2,5)))
dense.summary()
# %%
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
# %%
net.summary()