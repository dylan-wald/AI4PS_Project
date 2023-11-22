import numpy as np
import pandas as pd
from numpy import dot as dot

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
        X_train, y_train, X_test, y_test = X[:split,:], y[:split,:], X[split:,:], y[split:,:]

        return X_train, y_train, X_test, y_test




class PINNLoss(nn.Module):
    def __init__(self):
        super(PINNLoss, self).__init__()

        self.V_OP = 1000 # Volts
        self.V_ON = -1000 # Volts

        self.C1 = 1e9
        self.C2 = 1e9
        self.C3 = 1e9
        self.C4 = 1e9
        self.C5 = 1e9
        self.C6 = 1e9
        self.C7 = 1e9
        self.C8 = 1e9

        self.alpha = 1e-5

    def f(self, X_est, X_true):
        """
        state transition MMC dynamics
        """        
        # currents
        i_1 = X_true[:,-2]
        i_2 = X_true[:,-1]

        # switching (upper arm)
        S_1 = X_true[:,0]
        S_2 = X_true[:,1]
        S_3 = X_true[:,2]
        S_4 = X_true[:,3]

        # switching (lower arm)
        S_5 = X_true[:,4]
        S_6 = X_true[:,5]
        S_7 = X_true[:,6]
        S_8 = X_true[:,7]

        # current capacitence voltage
        Vc_k = X_est

        B = torch.tensor([
            [1/self.C1],
            [1/self.C2],
            [1/self.C3],
            [1/self.C4],
            [1/self.C5],
            [1/self.C6],
            [1/self.C7],
            [1/self.C8]
        ])
        # B = B.expand(-1,50)
        u = torch.vstack((
            i_1*(S_1 - 1),
            i_1*(S_2 - 1),
            i_1*(S_3 - 1),
            i_1*(S_4 - 1),
            -1*i_2*S_5,
            -1*i_2*S_6,
            -1*i_2*S_7,
            -1*i_2*S_8,
        ))

        self.Vc_k_1 = Vc_k + self.alpha*B*u

        return self.Vc_k_1

    def g(self, X_true):
        """
        MMC output dynamics
        """

        # switching (upper arm)
        S_1 = X_true[:,0]
        S_2 = X_true[:,1]
        S_3 = X_true[:,2]
        S_4 = X_true[:,3]

        # switching (lower arm)
        S_5 = X_true[:,4]
        S_6 = X_true[:,5]
        S_7 = X_true[:,6]
        S_8 = X_true[:,7]

        S_1_4 = torch.vstack((S_1, S_2, S_3, S_4))
        S_5_8 = torch.vstack((S_5, S_6, S_7, S_8))

        const = (self.V_OP - self.V_ON)/2
        upper_arm = (1/2)*torch.tensordot(torch.transpose(self.Vc_k_1[:4, :], 0,1), (1 - S_1_4))
        lower_arm = (1/2)*torch.tensordot(torch.transpose(self.Vc_k_1[4:, :], 0,1), S_5_8)
        self.V_th = const - upper_arm + lower_arm

        return self.V_th

    def L1(self, y_pred, y_true):

        # estimated Vth from NN
        Vth_k_1_NN = y_pred[:, -1]

        # true Vth from data
        Vth_k_1_true = y_true

        loss_1 = torch.mean((Vth_k_1_NN - Vth_k_1_true)**2)

        return loss_1

    def L2(self, y_pred, Vc_k_est, X_true):

        # estimated Vc from NN
        Vc_k_1_NN = torch.transpose(y_pred[:,:8],0,1)


        # estimated Vc from state dynamics f
        Vc_k_1_hat = self.f(Vc_k_est, X_true)

        # loss between Vc from NN and Vc from dynamical equations
        loss_2 = torch.norm(Vc_k_1_hat - Vc_k_1_NN, p=2)

        return loss_2

    def L3(self, y_pred, X_true):

        Vth_k_1_NN = y_pred[:,-1]

        Vth_k_1_hat = self.g(X_true)
        loss_3 = torch.mean((Vth_k_1_NN - Vth_k_1_hat)**2)

        return loss_3

    def forward(self, y_pred, y_true, X_true, Vc_k_est):

        # loss = self.L1(y_pred, y_true)
        # loss = self.L2(y_pred, Vc_k_est, X_true)
        # loss = self.L2(y_pred, Vc_k_est, X_true) + self.L3(y_pred, X_true)
        loss = self.L1(y_pred, y_true) + self.L2(y_pred, Vc_k_est, X_true) + self.L3(y_pred, X_true)

        return torch.transpose(y_pred[:,:8],0,1).detach(), loss