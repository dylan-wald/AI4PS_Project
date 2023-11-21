import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

# torch custom dataset class
class CustomDataset(Dataset):

    def __init__(self, inputs, labels):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        inputs = self.inputs[idx]
        return inputs, label



class Data():

    def __init__(self, file):

        self.file = file

    def ImportData(self):

        df = pd.read_csv(self.file)

        return df

    def Normalize(self, X):

        X = MinMaxScaler().fit_transform(X)

        return X

    def TrainTestSplit(self, X, y, frac):

        split = int(len(X)*frac)
        X_train, y_train, X_test, y_test = X[:split,:,:], y[:split,:], X[split:,:,:], y[split:,:]

        return X_train, y_train, X_test, y_test




def PINNLoss(self):
    def __init__(self):
        super(PINNLoss, self).__init__()

    def f(self, input, state, disturbance):
        """
        state transition MMC dynamics
        """

        output = 1##dynamics

        return output

    def g(self, input, state, disturbance):
        """
        MMC output dynamics
        """

        output = 1## dyanmics

        return output

    def L1(self, target, output):

        loss_func = nn.MSELoss()
        loss_1 = loss_func(target, output)

        return loss_1

    def L2(self, target, inputs):

        loss_func = nn.MSELoss()
        output = self.f(inputs)
        loss_2 = loss_func(target, output)

        return loss_2

    def L3(self, target, inputs):

        loss_func = nn.MSELoss()
        output = self.g(inputs)
        loss_3 = loss_func(target, output)

        return loss_3

    def forward(self, target, inputs):

        loss = self.L1(target, inputs) + self.L2(target, inputs) + self.L3(target, inputs)

        return loss