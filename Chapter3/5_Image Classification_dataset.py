# %%
import tensorflow as tf

mnist_train,mnist_test = tf.keras.datasets.fashion_mnist.load_data()

# %%
print(len(mnist_train[0]), len(mnist_test[0])) ## train / test length
print(mnist_train[0].shape) # 60000, 28, 28
# %%
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
# %%
import matplotlib.pyplot as plt
def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    figsize = (num_cols * scale, num_rows*scale)
    _,axes = plt.subplots(num_rows,num_cols,figsize=figsize)
    axes= axes.flatten()
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
# %%
X,y = mnist_train[0][:18],mnist_train[1][:18]
print(X.shape)
show_images(X,2,9,titles=get_fashion_mnist_labels(y))
# %%
import threading
batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices(
    mnist_train).batch(batch_size).shuffle(len(mnist_train[0]))

def load_data_fashion_mnist(batch_size,resize=None):
    mnist_train,mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    ## tf.cast = 부동소수점은 버림, boolean은 True = 1 ,False = 0
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
# %%
