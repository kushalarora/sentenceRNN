import numpy as np


def build_vocab(filename):
    vocab = [None]
    vocab_dict = {None: 0}
    S = []
    fin = open(filename)
    if fin is None:
        raise IOError("Filename %s not found" % filename)

    index = 1
    for line in fin:
        words = line.split()
        sentence = []
        for i, word in enumerate(words):
            if word not in vocab_dict:
                vocab_dict[word] = index
                vocab.append(word)
                index += 1
            idx = vocab_dict[word]
            sentence.append(idx)
        S.append(sentence)
    return (S, vocab, vocab_dict)


def index_data(filename, Vindex):
    fin = open(filename)
    if fin is None:
        raise IOError("Filename %s not found" % filename)

    if Vindex is None:
        raise BaseException("Vocab index not found")

    S = []
    for line in fin:

        words = line.split()
        sentence = []

        for word in words:
            try:
                id = Vindex[word]
            except KeyError:
                word = None
                id = 0;

            if id is None:
                raise RuntimeError(
                    "Word: %s missing in vocabulary" % word)

            sentence.append(id)

        S.append(sentence)

    return S
