# -*- coding: utf-8 -*-
from os import path, walk, listdir

class Corpus:
    """Collection of text documents. Corpus objects discover and store text
    files found in specified path. File search can be recursive, i.e. traversing
    also subfolders.

    Attributes:
        num_chars: Number of characters in corpus.
        num_words: Number of words in corpus.
    """

    def __init__(self):
        """Object initialization. """

        self._docs = []
        self.num_chars = 0
        self.num_words = 0

    def discover(self, root_dir, recursive=True):
        """Discovers text files and stores parsed content.

        Args:
            root_dir: Folder to discover.
            recursive: Traverses also subfolders in given path.

        Returns:
            Number of text files parsed.
        """

        text_files = []
        if recursive:
            for root, _, files in walk(root_dir):
                text_files = self._find_files(text_files, root, files)
        else:
            text_files = self._find_files(text_files, root_dir, listdir(root_dir))

        num_files = len(text_files)

        for fn in text_files:
            self.parse(fn)

        return num_files

    def parse(self, filepath):
        """Parses a text document."""

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        self.num_chars += len(text.replace(' ',''))
        self.num_words += len(text.split())

        self._docs.append(text)

    def _find_files(self, text_files, root, discovered):
        """Checks for actual text files to parse."""
        for fn in discovered:
            filepath = path.join(root, fn)
            if path.isfile(filepath) and fn.endswith('.txt'):
                text_files.append(filepath)

        return text_files

    def __iter__(self):
        self._iter_count = 0
        self.num_docs = len(self._docs)
        return self

    def __next__(self):
        if self._iter_count<self.num_docs:
            doc = self[self._iter_count]
            self._iter_count += 1
        else:
            raise StopIteration
        return doc

    def __len__(self):
        return len(self._docs)

    def __getitem__(self, key):
        """ Access a document from corpus. """
        try:
            return self._docs[key]
        except TypeError:
            raise TypeError('Document index must be an integer.')
        except IndexError:
            raise IndexError('Document not found at given index.')
