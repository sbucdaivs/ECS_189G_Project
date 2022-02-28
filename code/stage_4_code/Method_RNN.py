import torch
import torch.nn as nn
from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.base_class.method import method
import numpy as np


class Method_RNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """
    INPUT_DIM = 0  # len(TEXT.vocab)
    EMBEDDING_DIM = 500
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1

    def __init__(self, vocab_size: int):
        super().__init__()

        self.INPUT_DIM = vocab_size
        self.embedding = nn.Embedding(self.INPUT_DIM, self.EMBEDDING_DIM)
        self.rnn = nn.RNN(self.EMBEDDING_DIM, self.HIDDEN_DIM)
        self.fc = nn.Linear(self.HIDDEN_DIM, self.OUTPUT_DIM)
        self.sig = nn.Sigmoid()

    def forward(self, text):

        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        sig_out = self.sig(self.fc(hidden.squeeze(0)))
        return sig_out

    def train(self, train_loader):
        lr = 0.001
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing

        for e in range(epochs):
            print('epoch {} training...'.format(e))
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                predictions = self(inputs).squeeze(1)
                loss = criterion(predictions, labels.float())
                loss.backward()
                optimizer.step()

    #
    # def test(selfself, test_loader):


    # def init_hidden(self, batch_size):
    #     ''' Initializes hidden state '''
    #     # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
    #     # initialized to zero, for hidden state and cell state of LSTM
    #     weight = next(self.parameters()).data
    #
    #     hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
    #               weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
    #     return hidden

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train'])
        # print('--start testing...')
        # pred_y = self.test(self.data['test'])
        # return {'pred_y': pred_y, 'true_y': self.data['test']['y']}


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc
