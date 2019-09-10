# -*- coding: utf-8 -*-
from os import path, walk
from .dictionary import Dictionary

class Corpus:
    """Collection of text documents. Corpus objects discover and store files
    .txt found in specified path. File search is recursive, i.e. it traverses
    also subfolders.

    Attributes:
        top_dir: Top directory of txt documents.
        dictionary: Dictionary object which contains mapping of tokens.
    """

    def __init__(self, top_dir):
        """Object initialization. """

        self.top_dir = top_dir
        # create token map
        self.dictionary = Dictionary(self.iterate_documents(top_dir))

    def iterate_documents(self, top_dir):
        """Iterates over all txt files, yielding one document at a time."""

        for root, _, files in walk(self.top_dir):
            for fname in filter(lambda fname: fname.endswith('.txt'), files):
                with open(path.join(root, fname), 'r', encoding='utf-8') as file:
                    doc = file.read()

                yield doc

    def __iter__(self):
        """Implements iterable method."""
        for doc in self.iterate_documents(self.top_dir):
            yield doc
