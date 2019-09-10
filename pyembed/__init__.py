# -*- coding: utf-8 -*-
"""pyembed package.

Word embedding based on word2vec skip-gram algoritm.

"""
import pickle
from os import path
from logging.handlers import RotatingFileHandler
import logging
import numpy as np
from .datastore import DataStore
from .tokenizer import Tokenizer
from .dictionary import Dictionary
from .sampler import Sampler
from .corpus import Corpus
from .nn import NeuralNet

__author__ = "Andrea Capitanelli"
__license__ = "MIT"
__version__ = "1.1.0"
__maintainer__ = "Andrea Capitanelli"
__email__ = "andrea@capitanelli.gmail.com"
__status__ = "Prototype"

def get_logger():
    """Get streaming logger."""

    logger = logging.getLogger('word-embedding')

    # stream logging
    stream_log_formatter = logging.Formatter('%(asctime)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    stream_log_handler = logging.StreamHandler()
    stream_log_handler.setFormatter(stream_log_formatter)
    logger.addHandler(stream_log_handler)
    logger.setLevel('INFO')

    return logger

def draw_negative(unigram_table, unigram_table_size, negative):
    """Draws negative samples from the unigram table."""

    negative_labels = unigram_table[np.random.randint(0, high=unigram_table_size,
        size=negative, dtype='int')]

    return negative_labels

def train(data_dir, min_count=5, sample=0.001, win_size=5,
    dry_run=False, embedding_size=300, learning_rate=0.025, num_epochs=5000,
    negative=5, utable_size=100000000):
    """Main public function.

    Performs word embedding on a corpus of txt documents and saves results to a
    binary (pickled) file.

    Args:
        data_dir: Folder with documents of corpus to parse.
        win_size: Size of moving window for context words.
        dry_run: If true, loads corpus, generates and saves training samples
            without performing NN training.
        min_count: Minimum frequency for words to be sampled.
        sample: Scale factor for subsampling probability.
        embedding_size: Embedding size.
        learning_rate: NN learning rate for gradient descent.
        num_epochs: Num. of epochs to train.
        negative: Num. of negative samples.
        utable_size: Size of table for unigram distribution.

    """

    logger = get_logger()

    # generate training data
    # ----------------------------------------- #

    # parse corpus
    corpus = Corpus(data_dir)

    # generate training samples and save to disk
    sampler = Sampler(win_size, min_count, sample, datapath=data_dir)
    num_samples, discarded = sampler(corpus)
    gain = 100*discarded/(num_samples+discarded)
    logger.info(f'Generated {num_samples} samples, with {discarded} tokens discarded ({gain:.2f}%).')
    if dry_run:
        raise SystemExit(0)

    # load sampling data
    # ----------------------------------------- #

    # X/Y labels
    logger.info('Loading input/output labels...')
    X = DataStore.load(path.join(data_dir, *('samples', 'input.store')))
    Y = DataStore.load(path.join(data_dir, *('samples', 'output.store')))

    # dictionary
    logger.info('Loading dictionary...')
    token_store = DataStore.load(path.join(data_dir, *('samples',
        'dictionary.store')))
    dictionary = token_store[0]
    token_to_id = dictionary.token_to_id
    id_to_token = dictionary.id_to_token
    id_to_occur = dictionary.id_to_occur

    vocab_size = len(token_to_id)
    num_samples = len(X)

    # generate unigram table
    # ----------------------------------------- #
    if negative > 0:

        logger.info('Populating unigram table... ')
        unigram_table = []

        id_to_occur_2 = dict()
        for token_id, token_occur in id_to_occur.items():
            id_to_occur_2[token_id] = token_occur**(3/4)
        den = sum(id_to_occur_2.values())

        for token_id, token_occur in id_to_occur_2.items():
            num_items = int(token_occur/den*utable_size)
            unigram_table += [token_id for ii in range(num_items)]

        unigram_table = np.array(unigram_table)
        unigram_table_size = len(unigram_table)

    # training model
    # ----------------------------------------- #

    logger.info(f'Starting training with {vocab_size} words and {num_samples} samples...')

    neural_net = NeuralNet(vocab_size, embedding_size, learning_rate, negative)

    # fixed values
    if negative > 0:
        # one-hot encoded ground truth
        t_vector = np.zeros(negative+1, dtype='int')
        t_vector[0] = 1

    else:
        sampling_inds = None

    # define checkpoints
    checkpoints = set([k*round(num_samples/4) for k in [1, 2, 3, 4]])

    for epoch in range(num_epochs):

        logger.info(f'Starting epoch #{epoch}.')

        #epoch_cost = 0

        #for batch_idx,ii in enumerate(batch_inds):
        for ii in range(num_samples):

            input_label = X[ii]
            output_label = Y[ii]

            if negative > 0:

                # negative samples must be different from positive sample
                sampling_inds = np.zeros(negative+1, dtype='int')
                negative_labels = draw_negative(unigram_table,
                    unigram_table_size, negative)
                while np.any(np.equal(negative_labels, output_label)):
                    negative_labels = draw_negative(unigram_table,
                        unigram_table_size, negative)

                sampling_inds[0] = output_label
                sampling_inds[1:] = negative_labels

            else:

                # one-hot encoded ground truth
                t_vector = np.zeros(vocab_size, dtype='int')
                t_vector[output_label] = 1

            # forward propagation
            y_vector, h_vector = neural_net.forward_propagate(input_label, sampling_inds)

            # back propagation
            neural_net.back_propagate(input_label, t_vector, y_vector, h_vector, sampling_inds)

            if ii in checkpoints:

                # update learning rate
                neural_net.learning_rate *= 0.98

                # must find a proper way for evaluating epoch average cost
                #epoch_cost = - np.sum(t_vector * np.log(y_vector + 0.001), axis=0, keepdims=True)
                logger.info(f'Trained {ii} samples out of {num_samples} for epoch #{epoch}')

    logger.info('Training finished')

    # save a dict via pickle
    output = dict(token_to_id=token_to_id, id_to_token=id_to_token,
        word_embedding=neural_net.W1)

    with open('embedding', 'wb') as file:
        pickle.dump(output, file)

    logger.info('Embedding saved to file.')
