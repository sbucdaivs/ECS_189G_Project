import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.base_class.method import method
import numpy as np
from nltk.tokenize import word_tokenize

class Method_Generate(nn.Module):
    learning_rate = 0.001
    momentum = 0.9
    max_epoch = 10

    lstm_size = 128
    embedding_dim = 128
    num_layers = 3
    batch_size = 256

    def __init__(self, mName, mDescription, data_loader):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.data_loader = data_loader

        self.seq_len = self.data_loader.seq_len

        self.embedding = nn.Embedding(
            num_embeddings=self.data_loader.num_words,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, self.data_loader.num_words)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

    def train(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        dataloader = DataLoader(self.data_loader.get_dataset(), batch_size=self.batch_size)

        for epoch in range(self.max_epoch):
            state_h, state_c = self.init_state(self.seq_len)
            for batch, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                y_pred, (state_h, state_c) = self.forward(x, (state_h, state_c))
                loss = loss_function(y_pred.transpose(1, 2), y)
                state_h = state_h.detach()
                state_c = state_c.detach()
                loss.backward()
                optimizer.step()
            print({'epoch': epoch, 'loss': loss.item()})

            print(self.test("Knock knock. Who's there?", next_words=20))

    def test(self, text, next_words=100):
        words = word_tokenize(text.lower())
        state_h, state_c = self.init_state(len(words))
        for i in range(0, next_words):
            x = torch.tensor([[self.data_loader.word_to_index[w] for w in words[i:]]])
            y_pred, (state_h, state_c) = self.forward(x, (state_h, state_c))
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(self.data_loader.index_to_word[word_index])
        return words