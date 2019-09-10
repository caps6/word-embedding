# Word-embedding

An implementation of word2vec skip-gram algorithm for word embedding, with sub-sampling and negative sampling as in the origin implementation of [Mikolov paper](https://arxiv.org/abs/1301.3781).

The training function is organized as follows:
- corpus is generated parsing all *txt* files found in the specified folder
- training samples are generated and saved to disk
- in case of negative sampling, unigram table is generated too
- training process is started
- embedding data are saved to disk (with pickle)

Please note that code is written for convenience over performance and no specific optimization is in place, so embedding process for large corpora requires a very very long time!

### Quick Start

Clone and install:

```console
git clone https://github.com/acapitanelli/word-embedding.git
cd word-embedding
pip install .
```

From console:

```console
foo@bar:~$ pyembed -h
usage: pyembed [-h] [--win-size] [--dry-run] [--min-count] [--sample]
               [--embedding-size] [--learning-rate] [--epochs] [--negative]
               [--unigram-table]
               data_dir

An implementation of word2vec algorithm for word embedding.

positional arguments:
  data_dir           Folder with documents of corpus

optional arguments:
  -h, --help         show this help message and exit
  --win-size         Size of moving window for context words. (default: 5)
  --dry-run          If true, loads corpus, generates and saves training
                     samples without performing NN training (default: False)
  --min-count        Words appearing less than min-count are excluded from
                     corpus (default: 5)
  --sample           Scale factor for subsampling probability (default: 0.001)
  --embedding-size   Embedding size (default: 300)
  --learning-rate    NN learning rate for gradient descent (default: 0.025)
  --epochs           Num. epochs to train (default: 10)
  --negative         Num. of negative samples. Negative sampling is applied
                     only if greater than 0 (default: 5)
  --unigram-table    Size of table for unigram distribution (default:
                     100000000)
```
