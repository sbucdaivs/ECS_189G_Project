import torch
import torch.nn as nn
from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.base_class.method import method
import numpy as np


class Method_RNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """
    data = None
    # TODO: fix this later
    vocab_size = Dataset_Loader.vocab_size + 1
    vocab_size = 5048
    output_size = 1
    embedding_dim = 500
    hidden_dim = 256
    n_layers = 2
    dropout = nn.Dropout(0.3)

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # embedding and LSTM layers

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers,
                            dropout=0.5, batch_first=True)

        # linear and sigmoid layers
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def train(self, train_loader, test_loader):
        lr = 0.001
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing
        counter = 0
        print_every = 1
        clip = 5  # gradient clipping

        for e in range(epochs):
            # initialize hidden state
            h = self.init_hidden(2) # batch size =4

            # batch loop
            for inputs, labels in train_loader:
                counter += 1

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                optimizer.zero_grad()

                # get the output from the model
                inputs = inputs.type(torch.LongTensor)
                output, h = self.forward(inputs, h)

                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()

                if True :
                    # Get validation loss
                    val_h = self.init_hidden(2)
                    val_losses = []

                    for inputs, labels in test_loader:

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        inputs = inputs.type(torch.LongTensor)
                        output, val_h = self(inputs, val_h)
                        val_loss = criterion(output.squeeze(), labels.float())

                        val_losses.append(val_loss.item())

                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))



    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train'], self.data['test'])
        # print('--start testing...')
        # pred_y = self.test(self.data['test'])
        # return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
    # SentimentLSTM(
    #     (embedding): Embedding(74073, 400)
    # (lstm): LSTM(400, 256, num_layers=2, batch_first=True, dropout=0.5)
    # (dropout): Dropout(p=0.3)
    # (fc): Linear(in_features=256, out_features=1, bias=True)
    # (sig): Sigmoid()
    # )

    # def train(self):
    #     lr = 0.001
    #
    #     criterion = nn.BCELoss()
    #     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    #     # training params
    #
    #     epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing
    #
    #     counter = 0
    #     print_every = 100
    #     clip = 5  # gradient clipping
    #
    #
    #     net.train()
    #     # train for some number of epochs
    #     for e in range(epochs):
    #         # initialize hidden state
    #         h = net.init_hidden(batch_size)
    #
