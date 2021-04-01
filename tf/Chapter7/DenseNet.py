#%%
import tensorflow as tf
from func import *

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(filters=num_channels,kernel_size=(3,3),padding='same')
        self.listLayers = [self.bn,self.relu,self.conv]

    def call(self,x):
        y=x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y],axis=-1)
        return y
# %%
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self,num_convs,num_channels):
        super(DenseBlock,self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))
    def call(self,x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
# %%
blk = DenseBlock(2,10)
X = tf.random.uniform((4,8,8,3))
Y = blk(X)
Y.shape
# %%
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self,num_channels,**kwargs) :
        super(TransitionBlock,self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels,kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2,strides=2)

    def call(self,x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)

blk = TransitionBlock(10)
blk(Y).shape
# %%
def block_1():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64,kernel_size=7,strides=2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='same')
    ])
# %%
def block_2():
    net = block_1()
    num_channels, growth_rate = 64,32
    num_convs_in_dense_blocks = [4,4,4,4]

    for i,num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs,growth_rate))
        num_channels += num_convs * growth_rate
        if i !=len(num_convs_in_dense_blocks)-1:
            num_channels //=2
            net.add(TransitionBlock(num_channels))
    return net
# %%
def net():
    net = block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10))
    return net
# %%
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
train(net, train_iter, test_iter, num_epochs, lr)
# %%
