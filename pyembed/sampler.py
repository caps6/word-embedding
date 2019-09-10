# -*- coding: utf-8 -*-
from .tokenizer import Tokenizer
from .dictionary import Dictionary
from .datastore import DataStore
from os import path
import numpy as np

class Sampler:
    """Class for performing token sampling from corpus. """

    def __init__(self, min_count, win_size, sample, datapath=None):

        # win_size is related to half-window
        self.win_size = win_size

        # persisting samples
        self.datapath = datapath
        self.file_count = 1

        # sub-sampling threshold
        self.sample = sample
        self.min_count = min_count

    def __call__(self, corpus):
        """Samples corpus of documents. Extracts pairs of central and context words
        for embedding training.
        """

        dictionary = corpus.dictionary
        sample_path = path.join(self.datapath,'samples')

        tokenizer = Tokenizer(ascii_only=True)

        N = 1000000
        X = DataStore('input',  datapath=sample_path, block_size=N)
        Y = DataStore('output',  datapath=sample_path, block_size=N)
        token_store = DataStore('dictionary',  datapath=sample_path, block_size=N)

        num_samples = 0
        discarded = 0

        for doc in corpus:

            token_inds = [dictionary.token_to_id[token] for token in tokenizer(doc)]

            # pruning rare words
            token_inds, num_rare = self.prune(dictionary, token_inds)

            # subsample tokens
            token_inds, num_subsampled = self.subsample(dictionary, token_inds)

            # generate training pairs
            samples = self.moving_window(token_inds)

            doc_X, doc_Y = zip(*[(s[0], s[1]) for s in samples])
            X += doc_X
            Y += doc_Y

            num_samples += len(doc_X)
            discarded += num_rare + num_subsampled

        # save samples
        X.commit()
        Y.commit()

        # save token map
        token_store.add(dictionary)
        token_store.commit()

        return num_samples, discarded

    def prune(self, dictionary, token_inds):

        len_pre = len(token_inds)
        token_inds = [token_id for token_id in token_inds if dictionary.id_to_occur[token_id] > self.min_count]
        len_post = len(token_inds)
        discarded = len_pre - len_post

        return token_inds, discarded

    def subsample(self, dictionary, token_inds):
        """ Applies keeping probability in Dictionary object to decide whether
        to keep document tokens."""

        num_input_tokens = len(token_inds)

        # evaluate keeping probability for tokens
        num_tokens = len(dictionary.id_to_occur)
        token_frequencies = np.array([dictionary.id_to_occur[token_id]/num_tokens for token_id in token_inds])
        token_probs = (np.sqrt(token_frequencies / self.sample) + 1) * np.divide(self.sample, token_frequencies)

        s = np.random.uniform(size=len(token_inds))

        token_vec = np.array(token_inds)

        token_inds = list(token_vec[s < token_probs])
        num_output_tokens = len(token_inds)

        discarded = num_input_tokens - num_output_tokens

        return token_inds, discarded

    def moving_window(self, token_inds):
        """Applies a moving window to token list. """

        doc_samples = []

        win_size = self.win_size
        num_tokens = len(token_inds)

        for ii in range(num_tokens):

            central = token_inds[ii]
            context_inds = list(range(max(0, ii-win_size), ii)) + list(range(ii+1, min(num_tokens, ii+win_size+1)))

            doc_samples += [(central, token_inds[idx]) for idx in context_inds]

        return doc_samples
