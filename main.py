import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.utils import CustomDataset
from src.utils import Data
from src.utils import PINNLoss
from src.NN import FNN, LSTM

torch.manual_seed(101)


### data file
# file = "data/combinedDataRev3Labeled.csv"
file = "data/combinedDataRev4SlimLabeled.csv"

### create dataframe
DataObj = Data(file)
df = DataObj.ImportData()

### define input (X) and output (y) data
Vc_colnames = ["vc1","vc2","vc3","vc4","vc5","vc6","vc7","vc8"]
X_colnames = ["g1upper","g2upper","g3upper", "g4upper","g5upper","g6upper","g7upper","g8upper","i1","i2"]
y_colnames = ["vout"]
X = np.array(df[X_colnames])[:10000,:]
y = np.array(df[y_colnames])[1:10001,:]

# ### normalize the training data
X = DataObj.Normalize(X)
y = DataObj.Normalize(y)

### data DD = {(X,U,D), Y}
frac = 0.8
X_train, y_train, X_test, y_test = DataObj.TrainTestSplit(X, y, frac=frac)
X_train = X_train.reshape(-1, 1, 10)
y_train = y_train.reshape(-1, 1, 1)
X_test = X_test.reshape(-1, 1, 10)
y_test = y_test.reshape(-1, 1, 1)

### use the custom Pytorch Dataset generator to get train and test data
train_data = CustomDataset(X_train, y_train)
test_data = CustomDataset(X_test, y_test)

### use the generated train/test data and batch size to create DataLoader objects for the train and test (validataion) sets
batch_size = 10
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
valloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ### neural network parameters
in_dim = 10
hidden_dim = 512
out_dim = 9

### define the neural network
model = LSTM(in_dim, out_dim, hidden_dim)

### training parameters
num_epochs = 10
learn_rate = 0.0001

### Define the loss function
loss_func = PINNLoss()
# weight on nn output to true output loss
gamma1 = 1e0
# weight on nn state output to dynamic state output loss
gamma2 = 5e0
# weight on nn output to dynamic output loss
gamma3 = 5e5 #1e-10

### Define the optimizer
optimizer = torch.optim.Adam(model.parameters())

### initialize the state
Vc_k_est = torch.ones((batch_size, 1, 8))*1000

### Train the FNN model, monitor loss
loss_all = []
for i in range(num_epochs):
    l_tot = 0
    l1_tot = 0
    l2_tot = 0
    l3_tot = 0
    for X_train, y_train in trainloader:
    # for i in range(int(len(X_train)/batch_size)):

        ### TRAIN THE GENERATOR
        optimizer.zero_grad()

        y_pred = model(X_train)
        Vc_k_est, loss, l1, l2, l3 = loss_func(y_pred, y_train, X_train, Vc_k_est, gamma1, gamma2, gamma3)
        
        loss.backward()

        optimizer.step()

        l_tot = l_tot + loss.detach()
        l1_tot = l1_tot + l1.detach()
        l2_tot = l2_tot + l2.detach()
        l3_tot = l3_tot + l3.detach()

    print(f"Epech {i} loss: {l_tot} ({l1_tot}, {l2_tot}, {l3_tot})")
    loss_all.append(l_tot)



### Plot the train data
Vc_hat_train = []
Vth_hat_train = []
Vth_true_train = []
for X_train, y_train in  trainloader:

    y_pred_train = model(X_train).detach().numpy()

    Vc_hat_train.append(y_pred_train[:,:,:-1].reshape(-1,8))
    Vth_hat_train.append(y_pred_train[:,:,-1].flatten())

    Vth_true_train.append(y_train.detach().numpy().flatten())

Vc_hat_train = np.array(Vc_hat_train).reshape(-1,8)
Vc_true_train = np.array(df[Vc_colnames])[:8000,:]
Vth_hat_train = np.array(Vth_hat_train).flatten()
Vth_true_train =np.array(Vth_true_train).flatten()

fig, ax = plt.subplots(2,1)
for i in range(8):
    ax[0].plot(Vc_hat_train[:,i], label=f"Vc{i}")
    ax[0].plot(Vc_true_train[:,i], 'k-')

ax[1].plot(Vth_true_train, label="True")
ax[1].plot(Vth_hat_train, label="Est")
ax[1].legend()
plt.show()
    



### Plot the test data
Vc_hat = []
Vth_hat = []
Vth_true = []
for X_test, y_test in  valloader:

    y_pred_test = model(X_test).detach().numpy()

    Vc_hat.append(y_pred_test[:,:,:-1].reshape(-1,8))
    Vth_hat.append(y_pred_test[:,:,-1].flatten())

    Vth_true.append(y_test.detach().numpy().flatten())

Vc_hat = np.array(Vc_hat).reshape(-1,8)
Vc_true = np.array(df[Vc_colnames])[-2000:,:]
Vth_hat = np.array(Vth_hat).flatten()
Vth_true =np.array(Vth_true).flatten()

fig, ax = plt.subplots(2,1)
for i in range(8):
    ax[0].plot(Vc_hat[:,i], label=f"Vc{i}")
    ax[0].plot(Vc_true[:,i], 'k-')

ax[1].plot(Vth_true, label="True")
ax[1].plot(Vth_hat, label="Est")
ax[1].legend()
plt.show()
