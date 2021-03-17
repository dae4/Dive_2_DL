#%%
import tensorflow as tf
from tensorflow.python.ops.linalg_ops import norm
from func import *

batch_size, num_steps =32, 35
train_iter, vocab = load_data_time_machine(batch_size,num_steps)
def get_params(vocab_size,num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return tf.random.normal(shape)
    
    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                tf.zeros(num_hiddens))
    
    W_xz, W_hz, b_z = three()
    W_xr, W_hr, b_r = three()
    W_xh, W_hh, b_h = three()

    W_hq = normal((num_hiddens,num_outputs))
    b_q = tf.zeros(num_outputs)

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    return params


def init_gru_state(batch_size, num_hiddens):
    return(tf.zeros(shape=(batch_size,num_hiddens)),)

def gru(inputs,state,params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q= params
    H, = state
    outputs = []
    for X in inputs:
        Z = tf.keras.activations.sigmoid(tf.matmul(X,W_xz)+tf.matmul(H,W_hz)+b_z) 
        R = tf.keras.activations.sigmoid(tf.matmul(X,W_xr)+tf.matmul(H,W_hr)+b_r)
        H_tilda = tf.keras.activations.tanh(tf.matmul(X,W_xh)+tf.matmul(R*H,W_hh)+b_h)
        H = Z*H+(1-Z)*H_tilda
        Y = tf.matmul(H,W_hq)+b_q
        outputs.append(Y)
    return tf.concat(outputs,axis=0),(H,)

vocab_size, num_hiddens, device = len(vocab), 256, try_gpu()
num_epochs, lr = 500,1
device_name = try_gpu()._device_name
#%%
strategy = tf.distribute.OneDeviceStrategy(device_name)
model = RNNModelScratch(vocab_size=len(vocab),num_hiddens=num_hiddens,init_state=init_gru_state,forward_fn=gru)
params=get_params(len(vocab),num_hiddens)
#%%
train_rnn(model,train_iter,vocab,num_hiddens,lr,num_epochs,strategy,get_params, use_random_iter=True)
