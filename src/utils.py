import numpy as np
import pandas as pd
from numpy import dot as dot

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler

### TORCH CUSTOM DATASET CLASS
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


### MANUAL UTILITY FUNCTION CLASS
class Data():

    def __init__(self, file):

        self.file = file

    def ImportData(self):

        df = pd.read_csv(self.file)

        return df

    def Normalize(self, X):

        X = StandardScaler().fit_transform(X)

        return X

    def TrainTestSplit(self, X, y, frac):

        split = int(len(X)*frac)
        X_train, y_train, X_test, y_test = X[:split,:], y[:split,:], X[split:,:], y[split:,:]

        return X_train, y_train, X_test, y_test



### PHYSICS LOSS FUNCTION
class PINNLoss(nn.Module):
    def __init__(self, gamma2, gamma3):
        super(PINNLoss, self).__init__()

        # objective function weights
        self.gamma2 = gamma2
        self.gamma3 = gamma3

        # constant parameters
        self.V_OP = 1000 # Volts
        self.V_ON = 1000 # Volts

        self.C1 = 0.0014 #1e6
        self.C2 = 0.0014 #1e6
        self.C3 = 0.0014 #1e6
        self.C4 = 0.0014 #1e6
        self.C5 = 0.0014 #1e6
        self.C6 = 0.0014 #1e6
        self.C7 = 0.0014 #1e6
        self.C8 = 0.0014 #1e6

        self.alpha = 1e-5

    def f(self, X_est, X_true):
        """
        state transition MMC dynamics: Vc[k+1] = f_MMC(V[k],i[k],S[k])
        - X_est: Vc estimate from time k, i.e., Vc[k]
        - X_true: known inputs at time k, i.e., i_1[k], i_2[k], S_i[k]
        """        
        # currents
        i_1 = X_true[:,:,-2].reshape(-1,1,1)
        i_2 = X_true[:,:,-1].reshape(-1,1,1)

        # switching (upper arm)
        S_1 = X_true[:,:,0].reshape(-1,1,1)
        S_2 = X_true[:,:,1].reshape(-1,1,1)
        S_3 = X_true[:,:,2].reshape(-1,1,1)
        S_4 = X_true[:,:,3].reshape(-1,1,1)

        # switching (lower arm)
        S_5 = X_true[:,:,4].reshape(-1,1,1)
        S_6 = X_true[:,:,5].reshape(-1,1,1)
        S_7 = X_true[:,:,6].reshape(-1,1,1)
        S_8 = X_true[:,:,7].reshape(-1,1,1)

        # current capacitence voltage
        Vc_k = torch.transpose(X_est, 1,2)

        # capacitence matrix
        B = torch.diag(torch.tensor([1/self.C1, 1/self.C2, 1/self.C3, 1/self.C4, 1/self.C5, 1/self.C6, 1/self.C7, 1/self.C8]))
        B = B.expand(X_est.shape[0],X_est.shape[2],X_est.shape[2])

        # control action vector
        u = torch.stack((
            -1*i_1*S_1,
            -1*i_1*S_2,
            -1*i_1*S_3,
            -1*i_1*S_4,
            -1*i_2*S_5,
            -1*i_2*S_6,
            -1*i_2*S_7,
            -1*i_2*S_8),
        dim=1).reshape(-1,8,1)

        # calculate Vc[k+1]
        Vc_k_1 = Vc_k + self.alpha * torch.bmm(B, u)

        return Vc_k_1

    def g(self, y_pred, X_true):
        """
        MMC output dynamics Vth[k+1] = g_MMC(Vc[k+1], S_i[k])
        - y_pred: capacitence voltage predicted from LSTM (at time k+1)
        - X_true: known switching pattern at time k, i.e., S_i[k]
        """

        # switching pattern (upper arm)
        S_1 = X_true[:,:,0].reshape(-1,1,1)
        S_2 = X_true[:,:,1].reshape(-1,1,1)
        S_3 = X_true[:,:,2].reshape(-1,1,1)
        S_4 = X_true[:,:,3].reshape(-1,1,1)

        # switching pattern (lower arm)
        S_5 = X_true[:,:,4].reshape(-1,1,1)
        S_6 = X_true[:,:,5].reshape(-1,1,1)
        S_7 = X_true[:,:,6].reshape(-1,1,1)
        S_8 = X_true[:,:,7].reshape(-1,1,1)

        # combine into two vectors
        S_1_4 = torch.stack((S_1, S_2, S_3, S_4), dim=1).reshape(-1,4,1)
        S_5_8 = torch.stack((S_5, S_6, S_7, S_8), dim=1).reshape(-1,4,1)

        # compute the output voltage Vth
        const = (self.V_OP - self.V_ON)/2
        upper_arm = (1/2)*torch.tensordot(y_pred[:,:,:4], S_1_4).reshape(-1,1,1)
        lower_arm = (1/2)*torch.tensordot(y_pred[:,:,4:-1], S_5_8).reshape(-1,1,1)
        V_th = const - upper_arm + lower_arm

        return V_th

    def L2(self, y_pred, Vc_k_est, X_true):
        """
        State dynamics loss function
        - y_pred: predicted values from the LSTM
        - Vc_k_est: predicted Vc values from time k
        - X_true: known inputs at time k, i.e., i_1[k], i_2[k], S_i[k]
        """

        # estimated Vc from NN
        Vc_k_1_NN = torch.transpose(y_pred[:,:,:8],1,2)

        # estimated Vc from state dynamics f
        Vc_k_1_hat = self.f(Vc_k_est, X_true) / 1000

        # loss between Vc from NN and Vc from dynamical equations
        loss_2 = torch.mean((Vc_k_1_hat - Vc_k_1_NN)**2, dim=1) #+ 1e-8*torch.mean((Vc_k_1_NN - Vc_k_est)**2, dim=1)

        return loss_2

    def L3(self, y_pred, X_true):
        """
        Output dynamics loss function
        - y_pred: predicted values from the LSTM
        - X_true: known inputs at time k, i.e., i_1[k], i_2[k], S_i[k]
        """

        # estimated Vth from NN
        Vth_k_1_NN = y_pred[:,:,-1].reshape(-1,1,1)

        # estimated Vth from output equation
        Vth_k_1_hat = self.g(y_pred, X_true) / 1000

        # loss between Vth from NN and Vth from dynamics
        loss_3 = torch.mean((Vth_k_1_NN - Vth_k_1_hat)**2, dim=1)

        return loss_3

    def forward(self, y_pred, y_true, X_true, Vc_k_est):

        l2 = self.gamma2*self.L2(y_pred, Vc_k_est, X_true)
        l3 = self.gamma3*self.L3(y_pred, X_true)

        loss = l2 + l3

        return y_pred[:,:,:8].detach(), torch.mean(loss), torch.mean(l2), torch.mean(l3)