import torch
import torch.nn as nn
from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.base_class.method import method
import numpy as np

class Method_Generate(nn.Module):
    def __init__(self, mName, mDescription, num_words):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.num_words = num_words

        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        self.embedding = nn.Embedding(
            num_embeddings=self.num_words,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, num_words)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

    def train(self, data):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):

            state_h, state_c = model.init_state(args.sequence_length)

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = loss_function(y_pred.transpose(1, 2), y)
            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            optimizer.step()

    def test(self, X):
        pass

    def run(self):
        pass