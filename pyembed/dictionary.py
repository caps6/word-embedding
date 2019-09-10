# -*- coding: utf-8 -*-
import numpy as np
from .tokenizer import Tokenizer

class Dictionary:
    """ Counts tokens in a doc_iterator and keeps simple statistics.

    Attributes:
        token_to_id: Dict with tokens as keys and ids as their values.
        id_to_token: Dict with token ids as keys and tokens as values.
        id_to_occur: Dict with token ids as keys and occurrence frequencies as
            values.

    """

    def __init__(self, doc_iterator, sample=0.001):
        """Object initialization.

        Args:
            doc_iterator: Iterator for corpus documents.
            sample: Scae factor for keeping probability.
        """

        self.sample = sample
        self._tokenizer = Tokenizer(ascii_only=True)

        # init internal data
        token_to_id = dict()
        id_to_token = dict()
        id_to_occur = dict()

        token_idx = 0

        for doc in doc_iterator:

            # raw tokens
            tokens = self._tokenizer(doc)

            for tok in tokens:

                if tok in token_to_id:
                    token_id = token_to_id[tok]
                    id_to_occur[token_id] += 1

                else:
                    new_token_id = token_idx
                    token_to_id[tok] = new_token_id
                    id_to_token[new_token_id] = tok

                    id_to_occur[new_token_id] = 1
                    token_idx += 1

        # save
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.id_to_occur = id_to_occur
