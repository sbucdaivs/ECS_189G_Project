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

import os

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    vocab_size = 0

    def __init__(self, dName=None, dDescription=None, type='classification'):
        self.type = type
        super().__init__(dName, dDescription)

    all_text = [] # a list of all text, to count words and make vocab later
    def load_file(self, filename):
        file = open(filename, 'r')
        review = file.read()
        file.close()

        review = review.lower()
        review = ''.join([c for c in review if c not in punctuation])
        self.all_text.append(review)
        return review

    def pad_features(self, reviews_int, seq_length):
        '''
        Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
        '''
        features = np.zeros((len(reviews_int), seq_length), dtype=int)
        for i, review in enumerate(reviews_int):
            review_len = len(review)
            if review_len <= seq_length:
                zeroes = list(np.zeros(seq_length - review_len))
                new = zeroes + review
            elif review_len > seq_length:
                new = review[0:seq_length]
            features[i, :] = np.array(new)
        return features

    def load_classification(self):
        X_raw = []  # text
        X = []  # encoded
        y = []
        directory = self.dataset_source_folder_path
        for label_type in ['neg', 'pos']:
            data_folder = os.path.join(directory, label_type)
            for root, dirs, files in os.walk(data_folder):
                for fname in files:
                    if fname.endswith(".txt"):
                        file_name_with_full_path = os.path.join(root, fname)
                        clean_review = self.load_file(str(file_name_with_full_path))
                        X_raw.append(clean_review)
                        if label_type == 'neg':
                            y.append(0)
                        else:
                            y.append(1)
        all_text_2 = ' '.join(self.all_text)
        words = all_text_2.split()
        count_words = Counter(words)
        total_words = len(words)
        sorted_words = count_words.most_common(total_words)
        vocab_to_int = {w: i+1 for i, (w, c) in enumerate(sorted_words)} # start w/ 1, 0 for padding
        self.vocab_size = len(vocab_to_int)
        for review in X_raw: # encode text
            r = [vocab_to_int[w] for w in review.split()]
            X.append(r)
        reviews_len = [len(x) for x in X]
        X = [X[i] for i, l in enumerate(reviews_len) if l > 0]
        X = self.pad_features(X, 500)

        return {'X': X, 'y': y}

    def index_words(self, text):
        words = word_tokenize(text)
        word_counts = Counter(words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def load_generation(self):
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
        indexed = self.index_words(' '.join(data))
        self.num_words = len(indexed)

        self.index_to_word = {index: word for index, word in enumerate(indexed)}
        self.word_to_index = {word: index for index, word in enumerate(indexed)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

        for i in range():
            pass

        return data


    def load(self):
        print('loading data...')
        if self.type == 'classification':
            return self.load_classification()
        else:
            print('using generation loader')
            return self.load_generation()
