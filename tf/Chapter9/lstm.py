#%%
import tensorflow as tf
from func import *

batch_size, num_steps =32, 35
train_iter, vocab = load_data_time_machine(batch_size,num_steps)

def get_lstm_params(vocab_size,num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return tf.random.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32)
    
    def three():
        return (tf.Variable(normal((num_inputs,num_hiddens)),dtype=tf.float32),
                tf.Variable(normal((num_hiddens,num_hiddens)),dtype=tf.float32),
                tf.Variable(tf.zeros(num_hiddens),dtype=tf.float32))
    
    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()

    W_hq = tf.Variable(normal((num_hiddens,num_outputs)),dtype=tf.float32)
    b_q = tf.Variable(tf.zeros(num_outputs),dtype=tf.float32)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f,W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]

    return params

def init_lstm_state(batch_size, num_hiddens):
    return(tf.zeros(shape=(batch_size,num_hiddens)),tf.zeros(shape=(batch_size,num_hiddens)))

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f,W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params 
    H, C = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X,[-1,W_xi.shape[0]])
        I = tf.sigmoid(tf.matmul(X,W_xi)+tf.matmul(H,W_hi)+b_i)
        F = tf.sigmoid(tf.matmul(X,W_xf)+tf.matmul(H,W_hf)+b_f)
        O = tf.sigmoid(tf.matmul(X,W_xo)+tf.matmul(H,W_ho)+b_o)
        C_tilda = tf.tanh(tf.matmul(X,W_xc)+tf.matmul(H,W_hc)+b_c)
        C = F*C + I*C_tilda
        H = O * tf.tanh(C)
        Y = tf.matmul(H,W_hq) + b_q
        outputs.append(Y)

    return tf.concat(outputs,axis=0),(H,C)

vocab_size, num_hiddens, = len(vocab), 256
num_epochs, lr = 500,1
device_name = try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
model = RNNModelScratch( vocab_size, num_hiddens, init_lstm_state, lstm, get_lstm_params)
train_rnn(model, train_iter, vocab, lr, num_epochs, strategy, use_random_iter=True)
# %%
lstm_cell = tf.keras.layers.LSTMCell(num_hiddens, kernel_initializer='glorot_uniform')
lstm_layer = tf.keras.layers.RNN(lstm_cell, time_major=True, return_sequences=True, return_state=True)
with strategy.scope():
    model = RNNModel(lstm_layer, vocab_size=len(vocab))
train_rnn(model, train_iter, vocab, num_hiddens, lr, num_epochs, strategy)
# %%
