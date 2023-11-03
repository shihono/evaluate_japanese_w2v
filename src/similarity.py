# -*- coding: utf-8 -*-
import os
import csv
import logging
from collections import namedtuple
from chardet.universaldetector import UniversalDetector

import numpy as np
import gensim.models
from gensim.test.utils import datapath
from src import ja_tokenizer

"""handle similarity dataset"""
logger = logging.getLogger(__name__)


def detect_encoding(file_path):
    """
    search file encoding

    Parameters
    ----------
    file_path : str

    Returns
    -------
    encoding_name : str
    """
    detector = UniversalDetector()
    with open(file_path, 'rb') as f:
        for line in f:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    logger.info('detect encoding {}'.format(detector.result))
    return detector.result['encoding']


def load_keyvector(file_path):
    """
    load KeyedVectors

    Parameters
    ----------
    file_path : str
        file path
    Returns
    -------
    wv : KeyedVectors

    """
    _, ext = os.path.splitext(file_path)
    if ext == '.model':
        model = gensim.models.Word2Vec.load(file_path)
        wv = model.wv
        del model
        return wv
    elif ext == '.bin':
        wv = gensim.models.KeyedVectors.load_word2vec_format(datapath(file_path), binary=True)
        return wv
    elif ext == '.txt' or ext == '.vec':
        wv = gensim.models.KeyedVectors.load_word2vec_format(datapath(file_path), binary=False)
        return wv
    elif ext == '.kv':
        wv = gensim.models.KeyedVectors.load(file_path, mmap='r')
        return wv
    else:
        logger.warning("Cant load extension {} data".format(ext))
        return None


PairSim = namedtuple('PairSim', ['word1', 'word2', 'sim'])


class SimDataSet:
    def __init__(self, file_path, column_indexes=(0, 1, 2)):
        self.file_path = file_path
        self.column_indexes = column_indexes
        self.gold_data = self.load_csv(column_indexes=self.column_indexes)

    def load_csv(self, column_indexes):
        """
        load csv file

        Parameters
        ----------
        column_indexes: array-like
             word1, word2, similarityのインデックス

        Returns
        -------
        data : List
        """
        data = []
        enc = detect_encoding(self.file_path)
        with open(self.file_path, "r", encoding=enc) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                word1, word2, sim = [row[i] for i in column_indexes]
                data.append(PairSim(word1, word2, sim))
        logger.info('load {} data'.format(len(data)))
        return data

    def write_csv(self, res_array, save_path):
        with open(save_path, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            # header
            writer.writerow(['word1', 'word2', 'gold', 'pred'])
            for d, res in zip(self.gold_data, res_array):
                writer.writerow([d.word1, d.word2, res[0], res[1]])
        logger.info('save result into {}'.format(save_path))

    def __str__(self):
        return "filepath : {}, {} data".format(self.file_path, len(self.gold_data))


def cal_wv_similarity(dataset, wv, oov_score=-1, tokenizer=None):
    """
    calculate word similarity

    Parameters
    ----------
    dataset : SimDataSet obj
        evaluation dataset
    wv : `gensim.models.KeyedVectors`
        word2vec dictionary
    oov_score : int or None
         apply this score when the word is out of vocabulary in the model
    tokenizer : `JapaneseTokenizer` or None
        if word divide
    Returns
    -------
    result_array : ndarray
    """
    result_array = np.ones((len(dataset.gold_data), 2))

    oov_cnt = 0
    for i, d in enumerate(dataset.gold_data):
        word1, word2 = d.word1, d.word2
        if (word1 in wv.key_to_index) and (word2 in wv.key_to_index):
            sim = wv.similarity(word1, word2)
        else:
            if tokenizer is not None:
                sim = ja_tokenizer.get_similarity(word1, word2, wv, tokenizer)
                if sim is None:
                    logger.info('word {}, {} not in vocabulary'.format(word1, word2))
                    sim = oov_score
                    oov_cnt += 1
            else:
                logger.info('word {}, {} not in vocabulary'.format(word1, word2))
                sim = oov_score
                oov_cnt += 1

        result_array[i][0] = d.sim
        result_array[i][1] = sim
    logger.info('OOV cnt {}, {:.3%}'.format(oov_cnt, oov_cnt / len(dataset.gold_data)))
    return result_array
