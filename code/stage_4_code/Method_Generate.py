import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


class Method_Generate(nn.Module):
    learning_rate = 0.001
    momentum = 0.9
    max_epoch = 21

    lstm_size = 512
    embedding_dim = lstm_size
    num_layers = 3
    batch_size = 4096

    IS_LSTM = False

    def __init__(self, mName, mDescription, data_loader):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.data_loader = data_loader

        self.seq_len = self.data_loader.seq_len

        self.embedding = nn.Embedding(
            num_embeddings=self.data_loader.num_words,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.RNN(
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
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        # accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        dataloader = DataLoader(self.data_loader.get_dataset(), batch_size=self.batch_size)

        epoch_list = []
        loss_list = []

        for epoch in range(self.max_epoch):
            state_h, state_c = self.init_state(self.seq_len)
            scores = []
            for batch, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                if self.IS_LSTM:
                    y_pred, (state_h, state_c) = self.forward(x.to(device), (state_h.to(device), state_c.to(device)))
                else:
                    y_pred, state_h = self.forward(x.to(device), state_h.to(device))
                y_pred_cpu = y_pred.cpu()
                loss = loss_function(y_pred_cpu.transpose(1, 2), y)
                # print(y.detach().numpy().size())
                # print(y_pred_cpu.detach().numpy().size())
                # break
                # scores = precision_recall_fscore_support(y.detach().numpy(), y_pred_cpu.detach().numpy(), average='weighted')

                state_h = state_h.detach()
                state_c = state_c.detach()
                loss.backward()
                optimizer.step()

            epoch_list.append(epoch)
            loss_list.append(loss.item())
            if epoch % 10 == 0:
                # 'precision': scores[0], 'recall': scores[1], 'f1': scores[2]
                print({'epoch': epoch, 'loss': loss.item()})
                print(self.test("Knock knock. Who's there?", next_words=20))

        plt.plot(epoch_list, loss_list)
        plt.show()

    def test(self, text, next_words=100, device=device):
        words = word_tokenize(text.lower())
        state_h, state_c = self.init_state(len(words))
        for i in range(0, next_words):
            x = torch.tensor([[self.data_loader.word_to_index[w] for w in words[i:]]])
            if self.IS_LSTM:
                y_pred, (state_h, state_c) = self.forward(x.to(device), (state_h.to(device), state_c.to(device)))
            else:
                y_pred, state_h = self.forward(x.to(device), state_h.to(device))
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(self.data_loader.index_to_word[word_index])
        return ' '.join(words)