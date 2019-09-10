# -*- coding: utf-8 -*-
import re
import numpy as np

RE_TERMS = re.compile(r'\w+|\W')
RE_URL = re.compile(r"(?:(?:http|https):\/\/)?(?:www)?(?:\w+\.)+(?:\w+)\S+")

NON_ASCII = r' ,;.:-_@#()[]{}!?\'’^<>"“”£$%&/=+*\\|\n\t'

class Tokenizer:
    """Extracts tokens from text.

    By default tokens do not include non-ASCII characters.
    """

    def __init__(self, ascii_only=False):
        self._ascii_only = ascii_only

    def __call__(self, text):

        tokens = [m.group(0).lower() for m in RE_TERMS.finditer(text)]

        # remove non-ascii characters
        if self._ascii_only:
            tokens = list(filter(lambda w: w not in NON_ASCII, tokens))

        return tokens
