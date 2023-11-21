import numpy as np

import torch
from torch.utils.data import DataLoader

from src.utils import CustomDataset
from src.utils import Data
from src.utils import PINNLoss
from src.nn import FNN


### data file
file = "data/testData.csv"

### create dataframe
DataObj = Data(file)
df = DataObj.ImportData()

### define input (X) and output (y) data
X = np.array(df.X).reshape(-1,1)[:30000,:]
y = np.array(df.y).reshape(-1,1)[1:30001,:]

### normalize the training data
X = Data.Normalize(X)

### data DD = {(X,U,D), Y}
X_train, y_train, X_test, y_test = Data.TrainTestSplit(X, y)

### use the custom Pytorch Dataset generator to get train and test data
train_data = Data.CustomDataset(X_train, y_train)
test_data = Data.CustomDataset(X_test, y_test)

### use the generated train/test data and batch size to create DataLoader objects for the train and test (validataion) sets
batch_size = 5
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
valloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

### neural network parameters
in_dim = 1008
hidden_dim = 256
out_dim = 1

### define the neural network
model = FNN(in_dim, out_dim, hidden_dim)

### training parameters
num_epochs = 100
learn_rate = 0.0001

### Define the loss function
# loss_func = Data.PINNLoss()

### Define the optimizer
optimizer = torch.optim.Adam(model.parameters())

### Train the FNN model, monitor loss
loss_all = []
for i in range(num_epochs):
    l_tot = 0
    for X_train, y_train in  trainloader:
        
        ### TRAIN THE GENERATOR
        optimizer.zero_grad()

        y_pred = FNN(in_dim, out_dim, hidden_dim)
        loss = Data.PINNLoss(X_train, y_train)

        loss.backward()

        optimizer.step()

        l_tot = l_tot + loss.detach()

    print(f"Epech {i} loss: {l_tot}")
    loss_all.append(l_tot)


