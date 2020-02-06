# -*- coding: utf-8 -*-
import logging

import numpy as np
from numpy import dot
from gensim import matutils

import MeCab
from sudachipy import tokenizer
from sudachipy import dictionary

"""japanese word tokenizer"""
logger = logging.getLogger(__name__)


def load_mecab(dict_path=None):
    if dict_path is None:
        m = MeCab.Tagger('-Owakati')
    else:
        m = MeCab.Tagger('-Owakati -d {}'.format(dict_path))
    return m


def load_sudachi(mode=None):
    if mode is None:
        mode = tokenizer.Tokenizer.SplitMode.C
    else:
        mode = eval('tokenizer.Tokenizer.SplitMode.{}'.format(mode))
    t = dictionary.Dictionary().create(mode=mode)
    return t


class JapaneseTokenizer:
    def __init__(self, tokenizer_name='mecab',  **kwargs):
        self.tokenizer_name = tokenizer_name

        if self.tokenizer_name == 'sudachi':
            if 'mode' in kwargs.keys():
                self.tokenizer_obj = load_sudachi(mode=kwargs['mode'])
            else:
                self.tokenizer_obj = load_sudachi()

        else:  # mecab
            if 'dict_path' in kwargs:
                self.tokenizer_obj = load_mecab(dict_path=kwargs['dict_path'])
            else:
                self.tokenizer_obj = load_mecab()

    def divide_word(self, word):
        if self.tokenizer_name == 'mecab':
            res = self.tokenizer_obj.parse(word).strip().split()
            return res
        elif self.tokenizer_name == 'sudachi':
            res = [m.surface() for m in self.tokenizer_obj.tokenize(word)]
            return res


def get_divided_wv(word, wv, ja_tokenizer):
    if word in wv.vocab:
        return wv.get_vector(word)

    div_words = ja_tokenizer.divide_word(word)
    res_vectors = np.zeros((wv.vector_size, len(div_words)))
    logger.debug('{} divide into {}'.format(word, div_words))
    oov_cnt = 0
    for idx, w in enumerate(div_words):
        if w in wv.vocab:
            res_vectors[:, idx] = wv.get_vector(w)
        else:
            oov_cnt += 1

    if oov_cnt == len(div_words):
        return None
    logger.debug('OOV cnt {}/{}'.format(oov_cnt, len(div_words)))
    return res_vectors.sum(axis=1)


def get_similarity(word1, word2, wv, ja_tokenizer):
    """
    calculate similarity based on `WordEmbeddingsKeyedVectors.similarity`

    Parameters
    ----------
    word1 : str
    word2 : str
    wv : `KeyedVectors`
    ja_tokenizer : JapaneseTokenizer

    Returns
    -------
    sim : float or None
    """
    w1 = get_divided_wv(word1, wv, ja_tokenizer)
    w2 = get_divided_wv(word2, wv, ja_tokenizer)

    if (w1 is None) or (w2 is None):
        return None

    sim = dot(matutils.unitvec(w1), matutils.unitvec(w2))
    return sim
