# -*- coding: utf-8 -*-
"""pyembed package.

Word embedding based on word2vec skip-gram algoritm.

"""

import pickle
from datetime import datetime
#from sys import stdout
from os import path
import numpy as np
from .datastore import DataStore
from .tokenizer import Tokenizer
from .mapping import TokenMap
from .sampling import Sampler
from .corpus import Corpus
from .nn import NeuralNet

__author__ = "Andrea Capitanelli"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Andrea Capitanelli"
__email__ = "andrea@capitanelli.gmail.com"
__status__ = "Prototype"


def train(data_dir, recursive=True, sample_scale=0.001, win_size=5,
    dry_run=False, embedding_size=300, learning_rate=0.025, num_epochs=5000,
    batch_size=256, num_neg=5, utable_size=100000000):
    """Main public function.

    Performs word embedding on a corpus of documents and saves result to a
    binary file.

    Args:
        data_dir: Folder with documents of corpus to parse.
        recursive: If true, also subfolders are traversed. Default: true.
        sample_scale: Scale factor for subsampling probability.
        win_size: Size of moving window for context words.
        dry_run: If true, loads corpus, generates and saves training samples
            without performing NN training.
        sample_scale: Scale factor for subsampling probability.
        embedding_size: Embedding size.
        learning_rate: NN learning rate for gradient descent.
        num_epochs: Num. of epochs to train.
        batch_size: Size of batch for training samples.
        num_neg: Num. of negative samples.
        utable_size: Size of table for unigram distribution.

    """
    
    corpus = Corpus()
    num_files = corpus.discover(data_dir, recursive=recursive)
    print(f'Loaded {num_files} documents in corpus.')

    tokenizer = Tokenizer(ascii_only=True)
    token_map = TokenMap(corpus, tokenizer=tokenizer, sample_scale=sample_scale)

    # generate training samples and save to disk
    sampler = Sampler(tokenizer=tokenizer, token_map=token_map, datapath=data_dir, win_size=win_size)
    num_samples, discarded = sampler(corpus)
    gain = 100*discarded/(num_samples+discarded)
    print(f'Generated {num_samples} samples, with {discarded} tokens discarded ({gain:.2f}%).')

    if dry_run:
        raise SystemExit(0)

    # load sampling data
    # ----------------------------------------- #

    # token map
    print('Loading token map... ', end='')
    token_store = DataStore.load(path.join(data_dir,*('samples','tokenmap.store')))
    print('done.')

    # X labels
    print('Loading input labels... ', end='')
    X = DataStore.load(path.join(data_dir,*('samples','input.store')))
    print('done.')

    # Y labels
    print('Loading output labels... ', end='')
    Y = DataStore.load(path.join(data_dir,*('samples','output.store')))
    print('done.')

    token_map = token_store[0]

    token_to_id = token_map.token_to_id
    id_to_token = token_map.id_to_token
    id_to_occur = token_map.id_to_occur
    num_tokens = token_map.num_tokens

    vocab_size = len(token_to_id)
    num_samples = len(X)
    num_batches = int(np.floor(num_samples/batch_size))

    # generate unigram table
    # ----------------------------------------- #
    if num_neg>0:

        print('Populating unigram table... ', end='')
        unigram_table = []

        id_to_occur_2 = dict()
        for token_id, token_occur in id_to_occur.items():
            id_to_occur_2[token_id] = token_occur**(3/4)
        den = sum(id_to_occur_2.values())

        for token_id, token_occur in id_to_occur_2.items():
            n = int(token_occur/den*utable_size)
            unigram_table += [token_id for ii in range(n)]

        unigram_table = np.array(unigram_table)
        print('done.')


    # start training
    # ----------------------------------------- #

    print(f'Starting training with {vocab_size} words and {num_samples} samples...')

    neural_net = NeuralNet(vocab_size, embedding_size, learning_rate, num_neg,
        0.01)

    # start taking time
    begin_time = datetime.now()

    costs = []

    for epoch in range(num_epochs):

        batch_inds = list(range(0, num_samples, batch_size))

        np.random.shuffle(batch_inds)

        epoch_cost = 0

        for batch_idx,ii in enumerate(batch_inds):

            # load mini-batch samples
            X_batch = np.array(X[ii:ii+batch_size])
            Y_batch = np.array(Y[ii:ii+batch_size])

            if num_neg>0:

                # negative sampling
                unique_y_labels = list(set(Y_batch))
                num_pos = len(unique_y_labels)
                sampling_inds = np.zeros(num_pos+num_neg, dtype='int')
                sampling_inds[:num_pos] = np.array(unique_y_labels)
                sampling_inds[num_pos:] = unigram_table[np.random.randint(0, high=len(unigram_table), size=num_neg, dtype='int')]

            else:
                sampling_inds = np.arange(vocab_size)

            # Y one-hot
            num_cols = Y_batch.size
            Y_one_hot = np.zeros((vocab_size, num_cols))
            Y_one_hot[Y_batch.flatten(), np.arange(num_cols)] = 1
            
            # forward propagation
            softmax_out, word_vec = neural_net.propagate(X_batch, sampling_inds)

            # back propagation
            neural_net.back_propagate(X_batch, Y_one_hot, softmax_out, word_vec, sampling_inds)

            epoch_cost += -(1/num_cols) * np.sum(np.sum(Y_one_hot[sampling_inds, :] * np.log(softmax_out + 0.001), axis=0, keepdims=True), axis=1)
            costs.append(epoch_cost)

        if (num_epochs>=10 and epoch % round(num_epochs/10)) == 0:

            # update learning rate
            neural_net.learning_rate *= 0.98

            print(f'Cost at epoch {epoch}: {epoch_cost}')

    end_time = datetime.now()
    print('Training time: {}'.format(end_time - begin_time))

    # saving word embedding
    output = {
        'token_to_id': token_to_id,
        'id_to_token': id_to_token,
        'word_embedding': neural_net.wrd_emb
    }

    with open('embedding', 'wb') as f:
        pickle.dump(output, f)
    print('Embedding saved to file.')
