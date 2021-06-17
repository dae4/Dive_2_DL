#%%
import math
import torch
import os
import random
from func import *
import matplotlib.pyplot as plt

#@save
def read_ptb():
    data_dir = download_extract('ptb')
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]
#%%
sentences = read_ptb()
f'# sentences: {len(sentences)}'
# %%
vocab = Vocab(sentences, min_freq=10)
#%%
print(f'vocab size: {len(vocab)}')
# %%
def subsampling(sentences, vocab):
    # Map low frequency words into <unk>
    sentences = [[vocab.idx_to_token[vocab[tk]] for tk in line]
                 for line in sentences]
    # Count the frequency for each word
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    # Return True if to keep this token during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    # Now do the subsampling
    return [[tk for tk in line if keep(tk)] for line in sentences]

subsampled = subsampling(sentences, vocab)
# %%
set_figsize()
plt.hist([[len(line) for line in sentences],
              [len(line) for line in subsampled]])
plt.xlabel('# tokens per sentence')
plt.ylabel('count')
plt.legend(['origin', 'subsampled']);
# %%
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([line.count(token) for line in sentences])}, '
            f'after={sum([line.count(token) for line in subsampled])}')

compare_counts('the')
# %%
compare_counts('join')

# %%
corpus = [vocab[line] for line in subsampled]
corpus[0:3]
# %%
