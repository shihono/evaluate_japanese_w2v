# evaluate_japanese_w2v

日本語類似度評価データセットをword2vecモデルに適用するためのスクリプト


[mecab-python3](https://pypi.org/project/mecab-python3/) と [SudachiPy](https://pypi.org/project/SudachiPy/) による分かち書きに対応

## Requirement

- chardet
- numpy
- scipy
- gensim
- mecab-python3
- sudachipy
- sudachidict-core

## Usage

```bash
$ python eval.py model data [option]
```

- `model`: [gensim](https://radimrehurek.com/gensim/)で読み込み可能なモデルファイル
- `data`: 単語1, 単語2, (類似度などの)数値の3つの列を持つcsvファイルもしくはcsvファイルを含むディレクトリ
    - `--col` で3つの列を指定可能 (デフォルトは `[0,1,2]`)

```
optional arguments:
  -h, --help            show this help message and exit
  --col COL COL COL     indexes of word1, word2, similarity
  --verbose, -v         verbose
  --mecab, -m           use mecab
  --mecab_dict MECAB_DICT, -d MECAB_DICT
                        mecab dictionary path
  --sudachi, -s         use sudachi
  --sudachi_mode SUDACHI_MODE
                        select sudachi tokenizer mode: A or B or C
  --output OUTPUT, -o OUTPUT
                        output csv path or directory path
```


## Example

Example for Mecab

- model: [Japanese Word2Vec Model Builder](https://github.com/shiroyagicorp/japanese-word2vec-model-builder)
- data: [JWSAN](http://www.utm.inf.uec.ac.jp/JWSAN/) (similarity)
- tokenizer(optional): Mecab with [mecab-ipadic-neologd](https://github.com/neologd/mecab-ipadic-neologd)

```bash
$ python eval.py /path/to/latest-ja-word2vec-gensim-model/word2vec.gensim.model /path/to/JWSAN/jwsan-1400.txt \
    -v --col 1 2 4 -m --mecab_dict /usr/local/lib/mecab/dic/mecab-ipadic-neologd 
```

Output:

```
[XXXX-XX-XX XX:XX:XX,XXX] [__main__] [INFO] set logger
[XXXX-XX-XX XX:XX:XX,XXX] [__main__] [INFO] Word vector 50 dim, Vocab size 335476
[XXXX-XX-XX XX:XX:XX,XXX] [__main__] [INFO] Use mecab : dict setting is /usr/local/lib/mecab/dic/mecab-ipadic-neologd
[XXXX-XX-XX XX:XX:XX,XXX] [__main__] [INFO] load filepath : /path/to/JWSAN/jwsan-1400.csv, 1400 data
[XXXX-XX-XX XX:XX:XX,XXX] [__main__] [INFO] Evaluate 1359 data
[XXXX-XX-XX XX:XX:XX,XXX] [__main__] [INFO] spearmanr SpearmanrResult(correlation=0.4155930561711437, pvalue=6.97399627506598e-58)
Data    1400
OOV     41
Corr    0.416
```