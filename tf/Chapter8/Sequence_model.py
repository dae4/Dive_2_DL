#%%
import tensorflow as tf
import matplotlib.pyplot as plt

T=1000
time = tf.range(1,T+1,dtype=tf.float32)
x = tf.sin(0.01*time) + tf.random.normal([T],0,0.2)
plt.plot(time,x)
plt.xlabel('time')
plt.ylabel('x')
plt.xlim([1,1000])
# %%
tau = 4
features = tf.Variable(tf.zeros((T-tau,tau)))
for i in range(tau):
    features[:,i].assign(x[i:T-tau+i])

labels = tf.reshape(x[tau:],(-1,1))
# %%
def load_array(data_arrays,batch_size,is_train=True):
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset

batch_size, n_train = 16,600
train_iter = load_array((features[:n_train],labels[:n_train]),batch_size,is_train=True)

def get_net():
    net = tf.keras.Sequential([
        tf.keras.layers.Dense(10,activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return net

loss = tf.losses.MeanSquaredError()
# %%
class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset."""
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        l = loss(net(X), y)
        metric.add(tf.reduce_sum(l), tf.size(l))
    return metric[0] / metric[1]

def train(net,train_iter,loss,epochs,lr):
    trainer = tf.keras.optimizers.Adam()
    for epoch in range(epochs):
        for X,y in train_iter:
            with tf.GradientTape() as g:
                out= net(X)
                l = loss(y,out) /2
                params = net.trainable_variables
                grads = g.gradient(l,params)
            trainer.apply_gradients(zip(grads,params))
        print(f'epoch{epoch+1},'f'loss:{evaluate_loss(net,train_iter,loss):f}')
net = get_net()
train(net, train_iter,loss,5,0.01)
# %%
## prediction

onestep_preds = net(features)
plt.plot(time, x)
plt.plot(time[tau:],onestep_preds)
plt.xlabel('time')
plt.ylabel('x')
plt.xlim([1,1000])
plt.legend(['data', '1-step preds'])
# %%
multistep_preds = tf.Variable(tf.zeros(T))
multistep_preds[:n_train+tau].assign(x[:n_train+tau])
for i in range(n_train+tau,T):
    multistep_preds[i].assign(tf.reshape(net(tf.reshape(multistep_preds[i - tau: i], (1, -1))), ()))
plt.plot(time, x)
plt.plot(time[tau:],onestep_preds,'--')
plt.plot(time[n_train+tau],multistep_preds[n_train+tau],'-.')
plt.xlabel('time')
plt.ylabel('x')
plt.xlim([1,1000])
plt.legend(['data', '1-step preds','multistep preds'])
# %%
max_steps = 64
features = tf.Variable(tf.zeros((T - tau - max_steps + 1, tau + max_steps)))
for i in range(tau):
    features[:, i].assign(x[i: i + T - tau - max_steps + 1])
for i in range(tau, tau + max_steps):
    features[:, i].assign(tf.reshape(net((features[:, i - tau: i])), [-1]))

steps = (1, 4, 16, 64)
legends=[]

for i in steps:
    plt.plot(time[tau + i - 1:T - max_steps + i],features[:, (tau + i - 1)])
    legends.append(f'{i}-step preds')
plt.xlabel('time')
plt.ylabel('x')
plt.xlim([5,1000])
plt.legend(legends)