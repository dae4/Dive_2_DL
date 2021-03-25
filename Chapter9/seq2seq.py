#%%
import tensorflow as tf
import numpy
import math
from func import *
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        '''
        vocab_size: number of unique words
        embedding_dim: dimension of your embedding output
        enc_units: how many units of RNN cell
        batch_sz: batch of data passed to the training in each epoch
        '''
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
#%%
encoder = Encoder(vocab_size=10, embedding_dim=8, enc_units=16, batch_sz=4)
sample_hidden = encoder.initialize_hidden_state()
# print(sample_hidden)
X = tf.zeros((4, 7))
output, state = encoder(X, sample_hidden)
output.shape

#%%

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)


    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def call(self, x, state):
        # enc_output는 (batch_size, max_length, hidden_size)쌍으로 이루어져 있습니다.
        print(x)
        # 임베딩층을 통과한 후 x는 (batch_size, 1, embedding_dim)쌍으로 이루어져 있습니다.
        x = self.embedding(x)
        x = tf.transpose(x)
        context = state
        print(context)
        context_vector = np.broadcast_to(context, (X.shape[0], context.shape[0], context.shape[1]))
        # x = tf.concat([tf.expand_dims(context_vector, 2), x], axis=-1)
        X_and_context = tf.concat((X, context_vector), 2)
        # 위에서 결합된 벡터를 GRU에 전달합니다.
        output, state = self.gru(X_and_context)

        # output은 (batch_size * 1, hidden_size)쌍으로 이루어져 있습니다.
        output = tf.reshape(output, (-1, output.shape[2]))
        output = tf.transpose(self.dense(output))
        # output은 (batch_size, vocab)쌍으로 이루어져 있습니다.
        x = self.fc(output)

        return x, state

# %%
decoder = Decoder(vocab_size=10, embedding_dim=8, dec_units=16, batch_sz=4)
state = decoder.init_state(encoder(X,sample_hidden))
print(state.shape)
output, state = decoder(X, state)
output.shape, state.shape
#%%

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)
#%%
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        #initial decoder input - SOS token
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
        
        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            # using teacher forcing - decoder input for next time step is the target of the current time step
            dec_input = tf.expand_dims(targ[:, t], 1)
            
    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    
    # calculate gradient with respect to the model's trainable variables
    # essentially autodiff is happening here
    gradients = tape.gradient(loss, variables)   

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss
  
  

## To execute the training process
  
EPOCHS = 100

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

#%%
def evaluate(sentence):

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden = decoder(dec_input,dec_hidden,enc_out)

        print(predictions[0])
        predicted_id = tf.argmax(predictions[0]).numpy()
        print(predicted_id)

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            #return result, sentence, attention_plot
            return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

  
    return result, sentence

def translate(sentence):
    result, sentence = evaluate(sentence)
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))