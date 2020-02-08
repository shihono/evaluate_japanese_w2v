# -*- coding: utf-8 -*-
import os
import pathlib
import logging
import argparse
from glob import glob

import numpy as np
import scipy.stats
from src.similarity import SimDataSet, load_keyvector, cal_wv_similarity
from src.ja_tokenizer import JapaneseTokenizer

"""eval w2v similarity"""
logger = logging.getLogger(__name__)


def set_logger():
    """
    set logging StreamHandler
    """
    # logger.handlers.clear()
    format_string = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(format_string)

    # stdout
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('set logger')


def resolve_path(filepath):
    p = pathlib.Path(filepath)
    if p.is_absolute():
        return p
    else:
        return p.resolve()


def main():
    parser = argparse.ArgumentParser(description='japanese word similarity evaluation')
    parser.add_argument('model', help='gensim model path: extension like `.bin` or  `.model` or `.txt` or `.kv')
    parser.add_argument('data', help='evaluation dataset csv path or directory')
    parser.add_argument('--col', nargs=3, default=[0, 1, 2], type=int, help='indexes of word1, word2, similarity')
    parser.add_argument('--verbose', '-v', help='verbose', action='store_true')
    parser.add_argument('--mecab', '-m', help='use mecab', action='store_true')
    parser.add_argument('--mecab_dict', '-d', help='mecab dictionary path')
    parser.add_argument('--sudachi', '-s', help='use sudachi', action='store_true')
    parser.add_argument('--sudachi_mode', help='select sudachi tokenizer mode: A or B or C')
    parser.add_argument('--output', '-o', help='output csv path or directory path')
    args = parser.parse_args()

    if args.verbose:
        set_logger()
    data_path = resolve_path(args.data)
    if data_path.is_dir():
        data_path = glob(os.path.join(data_path, '*.csv'))
    else:
        data_path = [data_path]

    model_path = str(resolve_path(args.model))
    column_indexes = args.col

    wv = load_keyvector(model_path)
    logger.info('Word vector {} dim, Vocab size {}'.format(wv.vector_size, len(wv.vocab)))

    # set tokenizer : mecab or sudachipy
    tokenizer = None
    if args.mecab:
        logger.info('Use mecab : dict setting is {}'.format(args.mecab_dict))
        tokenizer = JapaneseTokenizer(tokenizer_name='mecab', dict_path=args.mecab_dict)
    elif args.sudachi:
        logger.info('Use sudachipy : mode setting is {}'.format(args.sudachi_mode))
        tokenizer = JapaneseTokenizer(tokenizer_name='sudachi', mode=args.sudachi_mode)

    for data in data_path:
        sim_dataset = SimDataSet(data, column_indexes)
        logger.info('load {}'.format(sim_dataset))
        res_array = cal_wv_similarity(sim_dataset, wv, oov_score=np.nan, tokenizer=tokenizer)
        if args.output:
            output_path = resolve_path(args.output)
            if output_path.is_dir():
                output_path = os.path.join(str(output_path), os.path.basename(data))
            else:
                output_path = str(output_path)
            sim_dataset.write_csv(res_array, output_path)
            logger.info('save {}'.format(output_path))

        res_array = res_array[~np.isnan(res_array).any(axis=1)]
        logger.info('Evaluate {} data'.format(res_array.shape[0]))

        spearmanr_result = scipy.stats.spearmanr(res_array)
        logger.info('spearmanr {}'.format(spearmanr_result))
        print('Data\t{}\nOOV\t{}\nCorr\t{:.3f}'.format(len(sim_dataset.gold_data),
                                                       len(sim_dataset.gold_data)-res_array.shape[0],
                                                       spearmanr_result[0]))


if __name__ == '__main__':
    main()