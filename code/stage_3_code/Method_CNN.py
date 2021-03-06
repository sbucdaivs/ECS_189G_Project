'''
Concrete MethodModule class for a specific learning MethodModule
'''

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Method_CNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 300
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 0.001
    momentum = 0.9
    epoch_list = []
    loss_list = []

    # TODO: Change the below to cnn

    # it defines the the MLP model architecture, e.g., how many layers, size of variables in each layer, activation
    # function, etc. the size of the input/output portal of the model architecture should be consistent with our data
    # input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # self.fc_layer_1 = nn.Linear(784, 784)  # Was (4,4) -> this is hidden layer 1
        # # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # self.activation_func_1 = nn.Sigmoid()
        # # self.fc_layer_2 = nn.Linear(784, 392)  # was (4,2) -> hidden layer 2
        # # # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        # # self.activation_func_2 = nn.Sigmoid()  # TODO: activation function?
        # self.fc_layer_3 = nn.Linear(784, 10)  # -> output layer
        # # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # self.activation_func_3 = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # # hidden layer embeddings
        # h0 = self.activation_func_1(self.fc_layer_1(x))
        # # h1 = self.activation_func_2(self.fc_layer_2(h0))
        # # output layer result
        # # self.fc_layer_2(h) will be a nx2 tensor
        # # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        # y_pred = self.activation_func_3(self.fc_layer_3(h0))
        # return y_pred
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return x

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

        # check here for the nn.CrossEntropyLoss doc:
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself

        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer
            # .zero_grad.html
            optimizer.zero_grad()

            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)).unsqueeze(1))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            self.epoch_list.append(epoch)
            self.loss_list.append(train_loss.item())
            if epoch % 100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                scores = accuracy_evaluator.evaluate()
                print('Epoch:', epoch, 'Accuracy:', scores[0], 'Loss:', train_loss.item())
        # self.epoch_list = epoch_list
        # self.loss_list = loss_list
        # plt.plot(epoch_list, loss_list)
        # plt.show()

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
