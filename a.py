import sys
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
import string
import copy
import csv
import time
import numpy as np
from collections import Counter
from collections import defaultdict

data_dir = Path("~/gobi2/data/project").expanduser()
TEXT_LIMIT = 60
VOCAB_SIZE = 100000


def preprocess(filename):
    data_path = data_dir / f"{filename}.csv"
    reader = pd.read_csv(data_path, sep=";", lineterminator='\r', chunksize=1000000)
    total_size = 0
    for i, chunk in enumerate(reader):
        out_file = data_dir / f"{filename}_{i}.data"
        print(f"chunk #{i} size:", len(chunk), flush=True)
        total_size += len(chunk)
        joblib.dump(chunk, out_file)
    print("total_size:", total_size)


def index_word(word, word_index):
    _word = word.lower().translate(str.maketrans('', '', string.punctuation))
    if len(_word) == 0:
        return [0, 0.]
    try:
        return [2, float(_word)]
    except:
        try:
            return [word_index.get(_word) + 3, 0.]
        except:
            return [0, 0.]


def index_word_new(word, word_index):
    _word = word.lower().translate(str.maketrans('', '', string.punctuation))
    if len(_word) == 0:
        return [0, 0.]
    try:
        return [2, float(word)]
    except:
        if len(word) > 1:
            try:
                return [2, float(word[1:])]
            except:
                pass
        try:
            return [word_index.get(_word) + 3, 0.]
        except:
            return [0, 0.]


def clean_words(words):
    _words = []
    for word in words:
        _word = word.lower().translate(str.maketrans('', '', string.punctuation))
        try:
            float(_word)
        except:
            _words.append(_word)
    return _words


class DatasetBuilder:

    def __init__(self, filename, word_index=None, n=np.inf, name="default"):
        self.word_index = word_index
        self.n = n

        self.load_words(filename, name)

        # thresholds = [100, 50, 20, 10, 5, 2]
        # i_threshold = 0
        # for i, word in enumerate(self.uniq_words):
        #     word_count = self.word_counts.get(word)
        #     while i_threshold < len(thresholds) and word_count < thresholds[i_threshold]:
        #         print(thresholds[i_threshold], i, word, word_count)
        #         i_threshold += 1
        #     if i_threshold == len(thresholds):
        #         break

    def load_words(self, filename, name):
        count_words = self.word_index is None
        train_df = pd.read_csv(data_dir / f"{filename}.csv", sep=";", lineterminator='\r', chunksize=1000000)
        total_size = 0
        word_counts = Counter()
        cleaned_data = defaultdict(lambda: [])
        length_distribution = defaultdict(lambda: 0)
        cnt = 0
        for i, chunk in enumerate(train_df):
            _cleaned_data = defaultdict(lambda: [])
            st = time.time()
            if count_words:
                words = chunk['text'].str.cat(sep=' ').split()
                words = clean_words(words)
                word_counts.update(words)
                joblib.dump(word_counts, data_dir / f"word_counts-clean-{i}.obj")
            else:
                for _, row in chunk.iterrows():
                    if type(row['text']) != str:
                        continue
                    cnt += 1
                    if cnt > self.n:
                        x = np.random.randint(cnt)
                        if x >= self.n:
                            continue
                    # print("row", _i)
                    __cleaned_data = dict()
                    def add_data(k, v):
                        # _cleaned_data[k].append(str(v))
                        __cleaned_data[k] = v
                    for _k, v in row.items():
                        k = str(_k)
                        if k == 'user' or k == 'fullname':
                            add_data(k, str(v))
                        elif k == 'timestamp':
                            try:
                                d = datetime.strptime(str(v) + '00', '%Y-%m-%d %H:%M:%S%z')
                            except ValueError:
                                d = datetime.strptime(str(v), '%Y-%m-%d %H:%M:%S%z')
                            add_data(k, int(d.timestamp()))
                        elif k == 'id' or k == 'replies' or k == 'likes' or k == 'retweets':
                            add_data(k, int(v))
                        elif k == 'text':
                            add_data('text', v)
                            # cleaned_words_idx = np.zeros(TEXT_LIMIT, dtype=int)
                            # cleaned_words_v = np.zeros(TEXT_LIMIT, dtype=float)
                            # # print(v.split())
                            # words = v.split()
                            # for word_i, word in enumerate(words):
                            #     if word_i >= TEXT_LIMIT:
                            #         break
                            #     # print(word, index_word(word, self.word_index))
                            #     idx, v = index_word(word, self.word_index)
                            #     cleaned_words_idx[word_i] = idx
                            #     cleaned_words_v[word_i] = v
                            # if len(words) < TEXT_LIMIT:
                            #     cleaned_words_idx[len(words)] = 1
                            # # length_distribution[len(cleaned_words)] += 1
                            # add_data('text_idx', cleaned_words_idx)
                            # add_data('text_v', cleaned_words_v)
                    if cnt <= self.n:
                        for k, v in __cleaned_data.items():
                            cleaned_data[k].append(v)
                    else:
                        p = np.random.randint(self.n)
                        for k, v in __cleaned_data.items():
                            cleaned_data[k][p] = v
                # print(cleaned_data.keys())
                # print(cleaned_data['text'][0].shape)
                # for k, v in cleaned_data.items():
                #     print('\n', k, len(v))
                #     for _v in v:
                #         print(_v)
                

                # joblib.dump(dict(_cleaned_data), data_dir / f"cleaned_data-{i}.obj")
                # joblib.dump(dict(length_distribution), data_dir / f"length_distribution-{i}.obj")
            print(f"{time.time() - st:.2f}s")

            print(f"chunk #{i} size:", len(chunk), flush=True)
            total_size += len(chunk)
        print("total_size:", total_size)
        # self.word_counts = word_counts
        if count_words:
            pass
            # joblib.dump(self.word_counts, data_dir / "word_counts-clean.obj")
        else:
            # for k in ['id', 'timestamp', 'replies', 'likes', 'retweets', 'text_idx', 'text_v']:
            for k in ['id', 'timestamp', 'replies', 'likes', 'retweets']:
                cleaned_data[k] = np.array(cleaned_data[k])
                print(k, cleaned_data[k].shape, cleaned_data[k].dtype)

            joblib.dump(dict(cleaned_data), data_dir / f"raw_data-{name}.obj")

    def build_word_index(self):
        self.uniq_words = sorted(self.word_counts, key=self.word_counts.get, reverse=True)
        self.word_index = dict()
        for i, word in enumerate(self.uniq_words):
            self.word_index[word] = i
        

def main():
    seed = hash("//".join(sys.argv[1:4])) % 2**32
    print("seed:", seed)
    np.random.seed(seed)
    # print(price_checker.query(2016, 1, 1))
    # print(price_checker.query(2019, 3, 31))
    word_index = joblib.load(data_dir / "word_index-clean.obj")
    builder = DatasetBuilder(sys.argv[1], word_index, int(sys.argv[2]), sys.argv[3])
    # preprocess(sys.argv[1])


if __name__ == "__main__":
    main()