import torch
import torch.nn as nn
from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.base_class.method import method
import numpy as np

class Method_Generate(nn.Module):
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

    def forward(self, x, hidden):
        pass

    def train(self, train_loader, test_loader):
        pass

    def test(self, X):
        pass

    def run(self):
        pass