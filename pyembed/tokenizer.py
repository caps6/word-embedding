# -*- coding: utf-8 -*-
import numpy as np
import re

RE_TERMS = re.compile(r'\w+|\W')
RE_URL = re.compile(r"(?:(?:http|https):\/\/)?(?:www)?(?:\w+\.)+(?:\w+)\S+")

ASCII = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
NON_ASCII = r' ,;.:-_@#()[]{}!?\'’^<>"“”£$%&/=+*\\|\n\t'

def token_match(token):
    """ Matches . """

    m = RE_URL.match(token)

    return (m is not None)

class Tokenizer:
    """Extracts tokens from text. Tokens include ASCII strings and non-ASCII
    characters.
    """

    def __init__(self, stopwords=None, ascii_only=False):
        self._ascii_only = ascii_only
        self._stopwords = stopwords

    def __call__(self, text):

        tokens = [m.group(0).lower() for m in RE_TERMS.finditer(text)]

        # remove non-ascii characters
        if self._ascii_only:
            tokens = list(filter(lambda w: w.lower() not in NON_ASCII, tokens))

        # remove stopwords
        if self._stopwords:
            tokens = list(filter(lambda w: w.lower() not in self._stopwords, tokens))

        return tokens

    def _valid(self,token):

        valid = (token != BLANK)

        if valid and self._stopwords:
            valid = valid and token.lower() not in self._stopwords

        return valid
