#%%
import tensorflow as tf
from func import *

def AlexNet():
    return tf.keras.models.Sequential([
       tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),     
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),       
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)
    ])
#%%
X = tf.random.uniform((1, 224, 224, 1))
for layer in AlexNet().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape:\t', X.shape)

#%%
batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
train(AlexNet, train_iter, test_iter, num_epochs, lr)
# %%
