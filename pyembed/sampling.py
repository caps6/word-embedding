# -*- coding: utf-8 -*-
from .tokenizer import Tokenizer
from .mapping import TokenMap
from .datastore import DataStore
from os import path
import numpy as np

class Sampler:
    """ Extracts token samples from a corpus. """

    def __init__(self, tokenizer=None,  token_map=None, win_size=5,
        size_thr=10000000, datapath=None):

        self._tokenizer = tokenizer or Tokenizer(ascii_only=True)
        self._token_map = token_map or TokenMap(tokenizer=self._tokenizer)

        # win_size is related to half-window
        self.win_size = win_size

        # persisting samples
        self.size_thr = size_thr
        self.datapath = datapath
        self.file_count = 1

        # sub-sampling threshold

    def __call__(self, corpus):
        """
        Sample corpus of documents. Extracts pairs of central and context words
        for embedding training.
        """

        sample_path = path.join(self.datapath,'samples')

        tokenizer = self._tokenizer
        token_map = self._token_map

        N = 1000000
        X = DataStore('input',  datapath=sample_path, block_size=N)
        Y = DataStore('output',  datapath=sample_path, block_size=N)
        token_store = DataStore('tokenmap',  datapath=sample_path, block_size=N)

        num_samples = 0

        discarded = 0

        for doc in corpus:

            tokens = tokenizer(doc)

            # apply keeping probability
            tokens, doc_discard = self._subsample(token_map, tokens)
            discarded += doc_discard

            samples = self._moving_window(tokens)

            doc_X, doc_Y = self._mapping(samples, token_map)

            X += doc_X
            Y += doc_Y

            num_samples += len(doc_X)

        # save samples
        X.commit()
        Y.commit()

        # save token map
        token_store.add(token_map)
        token_store.commit()

        return num_samples, discarded

    def _subsample(self, token_map, tokens):
        """ Applies keeping probability in TokenMap object to decide whether to
        keep document tokens."""

        in_tokens = len(tokens)
        token_inds = [token_map.token_to_id[tok] for tok in tokens]

        # keeping probabilities
        token_probs = np.array([token_map.id_to_prob[token_id] for token_id in token_inds])

        s = np.random.uniform(size=len(tokens))

        token_vec = np.array(tokens)

        tokens = list(token_vec[s < token_probs])
        out_tokens = len(tokens)

        discarded = in_tokens-out_tokens

        return tokens, discarded

    def _moving_window(self, tokens):
        """ Apply a moving window to token list. """

        doc_samples = []

        win_size = self.win_size
        num_tokens = len(tokens)

        for ii in range(num_tokens):

            central = tokens[ii]
            context_inds = list(range(max(0, ii-win_size), ii)) + list(range(ii+1, min(num_tokens, ii+win_size+1)))

            doc_samples += [(central, tokens[idx]) for idx in context_inds]

        return doc_samples

    def _mapping(self, samples, token_map):

        token_to_id = token_map.token_to_id

        token_ids = zip(*[(token_to_id[s[0]],token_to_id[s[1]]) for s in samples])
        return token_ids
