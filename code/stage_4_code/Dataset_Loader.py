# from code.base_class.dataset import dataset
import os
import re
import string
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from collections import Counter
import torch
from torch.nn import RNN
from torch.utils.data import Dataset, DataLoader


def load_file(filename):
    file = open(filename, 'r')
    review = file.read()
    file.close()

    # cleaning method:
    # https://sofiadutta.github.io/datascience-ipynbs/pytorch/Sentiment-Analysis-using-PyTorch.html
    en_stops = set(stopwords.words('english'))
    clean = re.compile('<.*?>')
    review_without_tag = re.sub(clean, '', review)
    review_without_tag_and_url = re.sub(r"http\S+", "", review_without_tag)
    review_without_tag_and_url = re.sub(r"www\S+", "", review_without_tag)
    review_lowercase = review_without_tag_and_url.lower()
    list_of_words = word_tokenize(review_lowercase)
    list_of_words_without_punctuation = [
        ''.join(this_char for this_char in this_string if (this_char in string.ascii_lowercase)) for this_string in
        list_of_words]
    list_of_words_without_punctuation = list(filter(None, list_of_words_without_punctuation))
    filtered_word_list = [w for w in list_of_words_without_punctuation if w not in en_stops]
    return ' '.join(filtered_word_list)


def get_data(directory, vocab, train_or_test):
    review_dict = {'neg': [], 'pos': []}
    if train_or_test == "train":
        directory = os.path.join(directory + '/train')
    elif train_or_test == "train_small":
        directory = os.path.join(directory + '/train_small')
    elif train_or_test == "test_small":
        directory = os.path.join(directory + '/test_small')
    else:
        directory = os.path.join(directory + '/test')
    for label_type in ['neg', 'pos']:
        data_folder = os.path.join(directory, label_type)
        for root, dirs, files in os.walk(data_folder):
            for fname in files:
                if fname.endswith(".txt"):
                    file_name_with_full_path = os.path.join(root, fname)
                    clean_review = load_file(file_name_with_full_path)
                    if label_type == 'neg':
                        review_dict['neg'].append(clean_review)
                    else:
                        review_dict['pos'].append(clean_review)
                    vocab.update(clean_review.split())
    return review_dict


vocab = Counter()
directory = '../../data/stage_4_data/text_classification'
train_review_dict = get_data(directory, vocab, "train_small")
test_review_dict = get_data(directory, vocab, "test_small")
word_list = sorted(vocab, key = vocab.get, reverse = True)
vocab_to_int = {word:idx+1 for idx, word in enumerate(word_list)}
int_to_vocab = {idx:word for word, idx in vocab_to_int.items()}


class IMDBReviewDataset(Dataset):

    def __init__(self, review_dict, alphabet):

        self.data = review_dict
        self.labels = [x for x in review_dict.keys()]
        self.alphabet = alphabet

    def __len__(self):
        return sum([len(x) for x in self.data.values()])

    def __getitem__(self, idx):
        label = 0
        while idx >= len(self.data[self.labels[label]]):
            idx -= len(self.data[self.labels[label]])
            label += 1
        reviewText = self.data[self.labels[label]][idx]

        label_vec = torch.zeros((1), dtype=torch.long)
        label_vec[0] = label
        return self.reviewText2InputVec(reviewText), label

    def reviewText2InputVec(self, review_text):
        T = len(review_text)

        review_text_vec = torch.zeros((T), dtype=torch.long)
        encoded_review = []
        for pos, word in enumerate(review_text.split()):
            if word not in vocab_to_int.keys():
                """
                If word is not available in vocab_to_int dict puting 0 in that place
                """
                review_text_vec[pos] = 0
            else:
                review_text_vec[pos] = vocab_to_int[word]

        return review_text_vec


def pad_and_pack(batch):
    input_tensors = []
    labels = []
    lengths = []
    for x, y in batch:
        input_tensors.append(x)
        labels.append(y)
        lengths.append(x.shape[0])  # Assume shape is (T, *)
    longest = max(lengths)
    print(longest)
    # We need to pad all the inputs up to 'longest', and combine into a batch ourselves
    if len(input_tensors[0].shape) == 1:
        x_padded = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=False)
    else:
        raise Exception('Current implementation only supports (T) shaped data')

    x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_padded, lengths, batch_first=False, enforce_sorted=False)

    y_batched = torch.as_tensor(labels, dtype=torch.long)

    return x_packed, y_batched


train_dataset=IMDBReviewDataset(train_review_dict,vocab)
test_dataset=IMDBReviewDataset(test_review_dict,vocab)

train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, collate_fn=pad_and_pack)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_and_pack)
