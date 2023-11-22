import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.utils import CustomDataset
from src.utils import Data
from src.utils import PINNLoss
from src.NN import FNN


### data file
file = "data/combinedDataRev3Labeled.csv"

### create dataframe
DataObj = Data(file)
df = DataObj.ImportData()

### define input (X) and output (y) data
X_colnames = ["g1upper","g2upper","g3upper", "g4upper","g5lower","g6lower","g7lower","g8lower","i1","i2"]
y_colnames = ["vout"]
X = np.array(df[X_colnames])[:5000,:]
y = np.array(df[y_colnames])[1:5001,:]

# ### normalize the training data
X = DataObj.Normalize(X)

### data DD = {(X,U,D), Y}
frac = 0.8
X_train, y_train, X_test, y_test = DataObj.TrainTestSplit(X, y, frac=frac)

### use the custom Pytorch Dataset generator to get train and test data
train_data = CustomDataset(X_train, y_train)
test_data = CustomDataset(X_test, y_test)

### use the generated train/test data and batch size to create DataLoader objects for the train and test (validataion) sets
batch_size = 1
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
valloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ### neural network parameters
in_dim = 10
hidden_dim = 512
out_dim = 9

### define the neural network
model = FNN(in_dim, out_dim, hidden_dim)

### training parameters
num_epochs = 10
learn_rate = 0.001

### Define the loss function
loss_func = PINNLoss()

### Define the optimizer
optimizer = torch.optim.Adam(model.parameters())

### initialize the state
Vc_k_est = torch.tensor([
    [1000],
    [1000],
    [1000],
    [1000],
    [1000],
    [1000],
    [1000],
    [1000],
])

### Train the FNN model, monitor loss
loss_all = []
for i in range(num_epochs):
    l_tot = 0
    for X_train, y_train in  trainloader:
        
        ### TRAIN THE GENERATOR
        optimizer.zero_grad()

        y_pred = model(X_train)
        Vc_k_est, loss = loss_func(y_pred, y_train, X_train, Vc_k_est)
        

        loss.backward()

        optimizer.step()

        l_tot = l_tot + loss.detach()

    print(f"Epech {i} loss: {l_tot}")
    loss_all.append(l_tot)
    # print(Vc_k_est)
    
### Plot the test data
Vc_hat = []

Vth_hat = []
Vth_true = []
for X_test, y_test in  valloader:

    y_pred_test = model(X_test).detach().numpy()

    Vc_hat.append(y_pred_test[:,:-1])
    Vth_hat.append(y_pred_test[:,-1])

    Vth_true.append(y_test.detach().numpy().flatten()[0])

Vc_hat = np.array(Vc_hat).reshape(-1,8)
Vth_hat = np.array(Vth_hat).flatten()

fig, ax = plt.subplots(2,1)
for i in range(8):
    ax[0].plot(Vc_hat[:,i])

ax[1].plot(Vth_true, label="True")
ax[1].plot(Vth_hat, label="Est")
ax[1].legend()
plt.show()
