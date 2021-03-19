#%%
import tensorflow as tf
from tensorflow.python.ops.linalg_ops import norm
from func import *

batch_size, num_steps =32, 35
train_iter, vocab = load_data_time_machine(batch_size,num_steps)
def get_params(vocab_size,num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return tf.random.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32)
    
    def three():
        return (tf.Variable(normal((num_inputs,num_hiddens)),dtype=tf.float32),
                tf.Variable(normal((num_hiddens,num_hiddens)),dtype=tf.float32),
                tf.Variable(tf.zeros(num_hiddens),dtype=tf.float32))
    
    W_xz, W_hz, b_z = three()
    W_xr, W_hr, b_r = three()
    W_xh, W_hh, b_h = three()

    W_hq = tf.Variable(normal((num_hiddens,num_outputs)),dtype=tf.float32)
    b_q = tf.Variable(tf.zeros(num_outputs),dtype=tf.float32)

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    return params

def init_gru_state(batch_size, num_hiddens):
    return(tf.zeros(shape=(batch_size,num_hiddens)), )

def gru(inputs,state,params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q= params
    H, = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        Z = tf.sigmoid(tf.matmul(X,W_xz)+tf.matmul(H,W_hz)+b_z) 
        R = tf.sigmoid(tf.matmul(X,W_xr)+tf.matmul(H,W_hr)+b_r)
        H_tilda = tf.tanh(tf.matmul(X,W_xh)+tf.matmul(R*H,W_hh)+b_h)
        H = Z*H+(1-Z)*H_tilda
        Y = tf.matmul(H,W_hq)+b_q
        outputs.append(Y)
    return tf.concat(outputs,axis=0),(H,)

vocab_size, num_hiddens, = len(vocab), 256
num_epochs, lr = 500,1
device_name = try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
model = RNNModelScratch(len(vocab),num_hiddens,init_gru_state,gru,get_params)
train_rnn(model,train_iter, vocab, lr, num_epochs, strategy, use_random_iter=True)
# %%
