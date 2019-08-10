# -*- coding: utf-8 -*-
import numpy as np

'''
TokenMap
~~~~~~~~~
This module provides an object to extract and identify stopwords from a Corpus.
'''
from .tokenizer import Tokenizer
import numpy as np

class TokenMap:
    """ Counts tokens in a corpus and keeps simple statistics. """

    def __init__(self, corpus, tokenizer=None, sample_scale=0.001):
        """Object initialization.

        Args:
            corpus: Corpus object with document collection.
            tokenizer: Tokenizer to use for mapping. If not given, a default
                Tokenizer object will be created.
        """

        self.sample_scale = sample_scale
        self._tokenizer = tokenizer or Tokenizer(ascii_only=True)

        # init internal data
        token_to_id = dict()
        id_to_token = dict()
        id_to_occur = dict()
        id_to_prob = dict()

        num_tokens = 0

        token_idx = 0

        for doc in corpus:

            # raw tokens
            tokens = self._tokenizer(doc)
            num_tokens += len(tokens)

            # set of unique words
            #words = list(set(tokens))

            for tok in tokens:

                if tok in token_to_id.keys():
                    token_id = token_to_id[tok]
                    id_to_occur[token_id] += 1

                else:
                    new_token_id = token_idx
                    token_to_id[tok] = new_token_id
                    id_to_token[new_token_id] = tok

                    id_to_occur[new_token_id] = 1
                    token_idx += 1

        # normalize occurrences to frequency and evaluate keeping probability
        for token_id, token_occur in id_to_occur.items():
            id_to_prob[token_id] = min(1, self._eval_prob(token_occur/num_tokens))

        # save
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.id_to_occur = id_to_occur
        self.id_to_prob = id_to_prob
        self.num_tokens = num_tokens

    def _eval_prob(self, word_freq):
        """ Evaluates subsampling probability basing of occurrence frequency
        of word and a scale factor."""

        prob = (np.sqrt(word_freq/self.sample_scale)+1)*self.sample_scale/word_freq

        return prob
