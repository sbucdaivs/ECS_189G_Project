'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code_src.base_class.dataset import dataset
from string import punctuation
from collections import Counter
import numpy as np
import os


def pad_features(reviews_int, seq_length):
    '''
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype=int)
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length - review_len))
            # apply left padding
            new = zeroes + review
        elif review_len > seq_length:
            new = review[0:seq_length]
        features[i, :] = np.array(new)
    return features


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    vocab_size = 0
    all_text = []  # a list of all text, to count words and make vocab later

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_file(self, file_path):
        """
        :param file_path: file path to a review .txt file
        :return: a str that's lower-cased, punctuation-removed
        """
        file = open(file_path, 'r')
        review = file.read()
        file.close()
        review = review.lower()
        review = ''.join([c for c in review if c not in punctuation])
        self.all_text.append(review)
        return review

    def load(self):
        print('loading data...')
        X_raw = []  # text
        X = []  # encoded
        y = []
        directory = self.dataset_source_folder_path
        # for every type:
        for label_type in ['neg', 'pos']:
            data_folder = os.path.join(directory, label_type)
            # for every file in the neg/pos folder:
            for root, dirs, files in os.walk(data_folder):
                for fname in files:
                    # exclude non-text file
                    if fname.endswith(".txt"):
                        file_name_with_full_path = os.path.join(root, fname)
                        clean_review = self.load_file(str(file_name_with_full_path))
                        X_raw.append(clean_review)
                        if label_type == 'neg':
                            y.append(0)
                        else:
                            y.append(1)
        all_text_2: str = ' '.join(self.all_text) # a str that contains all words in data
        words = all_text_2.split()
        count_words = Counter(words)
        total_words = len(words)
        # TODO: count_words is already a sorted dictionary?
        sorted_words = count_words.most_common(total_words)
        vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}  # start w/ 1, 0 for padding
        self.vocab_size = len(vocab_to_int)
        for review in X_raw:  # encode text
            r = [vocab_to_int[w] for w in review.split()]
            X.append(r)
        reviews_len = [len(x) for x in X]
        X = [X[i] for i, l in enumerate(reviews_len) if l > 0] # what's this line for?
        X = pad_features(X, 500)
        # X = pad_features(X, 300)

        # TODO: shuffle data?
        return {'X': X, 'y': y}
