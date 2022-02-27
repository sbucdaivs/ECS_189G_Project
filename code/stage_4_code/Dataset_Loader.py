'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
from string import punctuation
from collections import Counter
import numpy as np
import csv
from nltk.tokenize import word_tokenize
import torch

import os

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    vocab_size = 0

    def __init__(self, dName=None, dDescription=None, type='classification', seq_len = 4):
        self.type = type
        self.num_words = None
        self.dataset = None
        self.seq_len = seq_len
        self.index_to_word = None
        self.word_to_index = None

        super().__init__(dName, dDescription)

    def index_words(self, words):
        word_counts = Counter(words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def load(self):
        X_raw = []  # text
        X = []  # encoded
        y = []
        directory = self.dataset_source_folder_path
        file_name_with_full_path = os.path.join(directory, 'data')

        data = []
        with open(file_name_with_full_path, newline='') as f:
            reader = csv.reader(f)
            for line in reader:
                data.append(line[1].lower())
        data = data[1:]
        # print(word_tokenize(data[0]))
        all_words = word_tokenize(' '.join(data))
        indexed = self.index_words(all_words)
        self.num_words = len(indexed)

        self.index_to_word = {index: word for index, word in enumerate(indexed)}
        self.word_to_index = {word: index for index, word in enumerate(indexed)}

        words_indexes = [self.word_to_index[w] for w in all_words]

        self.dataset = Dataset(self.seq_len, words_indexes)

        return data

    def get_dataset(self):
        if self.dataset is None:
            raise ValueError("dataset not loaded")
        else:
            return self.dataset


class Dataset(torch.utils.data.Dataset):
    def __init__( self, seq_len, words_indexes):
        self.words_indexes = words_indexes
        self.seq_len = seq_len

    def __len__(self):
        return len(self.words_indexes) - self.seq_len

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.seq_len]),
            torch.tensor(self.words_indexes[index+1:index+self.seq_len+1]),
        )