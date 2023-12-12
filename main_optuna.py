import numpy as np
import matplotlib.pyplot as plt
import optuna

import torch
from torch.utils.data import DataLoader

from src.utils import CustomDataset
from src.utils import Data
from src.utils import PINNLoss
from src.NN import FNN, LSTM

from sklearn.preprocessing import MinMaxScaler, StandardScaler

torch.manual_seed(101)


### data file
# file = "data/combinedDataRev3Labeled.csv"
# file = "data/combinedDataRev4SlimLabeled.csv"
# file = "data/combinedDataRev4LabeledSynthetic.csv"

# fault conditions
# file = "data/combinedSimpleFaultLabeled.csv" # all fault data
# file = "data/combinedSimpleNormalLabeled.csv" # no fault data
# file = "data/combinedSimpleFaultComboLabeled.csv" # combined fault and no fault data (fake)
# file = "data/completeDatasetLabeled.csv" # combined fault and no fault data (real)
file = "data/completeDataFaultSimpleV3Labeled.csv" # combined fault and no fault data (real - short fault)

### create dataframe
DataObj = Data(file)
df = DataObj.ImportData()

### define input (X) and output (y) data
Vc_colnames = ["vc1","vc2","vc3","vc4","vc5","vc6","vc7","vc8"]
X_colnames = ["g1upper","g2upper","g3upper", "g4upper","g5upper","g6upper","g7upper","g8upper","i1","i2"]
y_colnames = ["vout"]
X = np.array(df[X_colnames])#[5000:15000,:]
y = np.array(df[y_colnames])#[5001:15001,:]

Vc_all = np.array(df[Vc_colnames])#[5000:15000,:]

### data DD = {(X,U,D), Y}
frac = 0.8
X_train, y_train, X_test, y_test = DataObj.TrainTestSplit(X, y, frac=frac)
X_train = X_train.reshape(-1, 1, 10)
y_train = y_train.reshape(-1, 1, 1)
X_test = X_test.reshape(-1, 1, 10)
y_test = y_test.reshape(-1, 1, 1)

def objective(trial, X_train, y_train, X_test, y_test):

    batch_size = 200

    # ### neural network parameters
    in_dim = 10
    # hidden_dim = trial.suggest_int("hidden_layers", 64, 1024, log=True) #256
    hidden_dim = trial.suggest_int("hidden_layers", 100, 300, log=True)
    out_dim = 9

    ### define the neural network
    model = LSTM(in_dim, out_dim, hidden_dim)

    ### training parameters
    num_epochs = 100
    # learn_rate = trial.suggest_float("lr", 1e-6, 0.1, log=True) #0.00001
    learn_rate = trial.suggest_float("lr", 0.00813*0.1, 0.00813*10, log=True)

    ### Define the loss function
    # weight on nn output to true output loss
    # gamma1 = trial.suggest_float("gamma1", 1e-4, 10.0, log=True) #.1 #1.0 / 47260.1495314765 #5e-6
    gamma1 = trial.suggest_float("gamma1", 1.54*0.1, 1.54*10, log=True)

    # weight on nn state output to dynamic state output loss
    # gamma2 = trial.suggest_float("gamma2", 1e-4, 10.0, log=True) #10.0 #10. #1e0 / 2.899061760923799
    gamma2 = trial.suggest_float("gamma2", 0.844*0.1, 0.844*10, log=True)

    # weight on nn output to dynamic output loss
    # gamma3 = trial.suggest_float("gamma3", 1e-4, 10.0, log=True) #.1 #/ 47260.1495314765 #5e-6
    gamma3 = trial.suggest_float("gamma3", 0.00058*0.1, 0.00058*10, log=True)

    physics_loss_func = PINNLoss(gamma2, gamma3)
    loss_func = torch.nn.MSELoss()

    ### Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    ### initialize the state 
    # Vc_k_est = torch.tensor(Vc_all[:batch_size, :], requires_grad=True, dtype=torch.float32).reshape(batch_size, 1, 8)

    idx = np.arange(0,len(X_train),batch_size)
    # idx = np.linspace(0,len(X_train),int(len(X_train)/batch_size)+1).astype(int)
    ### Train the FNN model, monitor loss
    loss_all = []
    for j in range(num_epochs):
        l_tot = 0
        l1_tot = 0
        l2_tot = 0
        l3_tot = 0
        # for X_train, y_train in trainloader:
        for i in range(len(idx)-1):

            Vc_k_est = torch.tensor(Vc_all[idx[i]:idx[i+1], :], dtype=torch.float32).reshape(batch_size, 1, 8)

            X_train_torch = torch.tensor(X_train[idx[i]:idx[i+1], :, :], dtype=torch.float32) / 10
            y_train_torch = torch.tensor(y_train[idx[i]:idx[i+1], :, :], dtype=torch.float32) / 1000

            optimizer.zero_grad()

            y_pred = model(X_train_torch)

            mse_loss = gamma1*loss_func(y_train_torch, y_pred[:, :, -1].reshape(-1, 1, 1))
            Vc_k_est, physics_loss, l2, l3 = physics_loss_func(y_pred, y_train_torch, X_train_torch*10, Vc_k_est)
            loss = mse_loss + physics_loss
            
            loss.backward()

            optimizer.step()

            l_tot = l_tot + loss.detach()
            l1_tot = l1_tot + mse_loss.detach()
            l2_tot = l2_tot + l2.detach()
            l3_tot = l3_tot + l3.detach()

        # print(f"Epoch {j} loss: {l_tot} ({l1_tot}, {l2_tot}, {l3_tot})")
        loss_all.append(l_tot)

    ###Calculate training error
    train_err = 0
    # for X_train, y_train in  trainloader:
    for i in range(len(idx)-1):

        X_train_torch = torch.tensor(X_train[idx[i]:idx[i+1], :, :], dtype=torch.float32) / 10
        y_train_torch = torch.tensor(y_train[idx[i]:idx[i+1], :, :], dtype=torch.float32)
        Vc_k_est = torch.tensor(Vc_all[idx[i]:idx[i+1], :], dtype=torch.float32).reshape(batch_size, 8)

        y_pred_train = model(X_train_torch/10).detach().numpy()*1000

        train_err = train_err + np.mean((Vc_k_est.detach().numpy() - y_pred_train[:,:,:-1].reshape(-1,8))**2) / batch_size
        train_err = train_err + np.mean((y_train_torch.detach().numpy().flatten() - y_pred_train[:,:,-1].flatten())**2) / batch_size


    # ### Calculate testing error
    # idx = np.linspace(len(X_train),len(X_train)+len(X_test),int(len(X_test)/batch_size)+1).astype(int)
    # test_err = 0
    # # for X_test, y_test in  valloader:
    # for i in range(len(idx)-1):

    #     X_test_torch = torch.tensor(X_test[idx[i]:idx[i+1], :, :], dtype=torch.float32) / 10
    #     y_test_torch = torch.tensor(y_test[idx[i]:idx[i+1], :, :], dtype=torch.float32)
    #     Vc_k_est = torch.tensor(Vc_all[idx[i]:idx[i+1], :], dtype=torch.float32).reshape(batch_size, 8)

    #     y_pred_test = model(X_test_torch).detach().numpy()*1000

    #     test_err = test_err + np.mean((Vc_k_est.detach().numpy() - y_pred_test[:,:,:-1].reshape(-1,8))**2) / batch_size
    #     test_err = test_err + np.mean((y_test_torch.detach().numpy().flatten() - y_pred_test[:,:,-1].flatten())**2) / batch_size

    return train_err #np.mean([train_err, test_err])


def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
            frozen_trial.number,
            frozen_trial.value,
            frozen_trial.params,
            )
        )

### Turn off optuna log notes
optuna.logging.set_verbosity(optuna.logging.WARN)

### Create the Optuna study (minimize mse)
study = optuna.create_study(direction="minimize")

### Run the Optuna study (only log trials that improve the objective)
study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50, callbacks=[logging_callback])

print("Best trial:")
trial = study.best_trial

print(f"  Accuracy: {trial.value} %")

print("  Optimal parameters: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))