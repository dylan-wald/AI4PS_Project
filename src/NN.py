import torch.nn as nn

### FULLY CONNECTED NEURAL NETWORK CLASS
class FNN(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.linear_3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.linear_4 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self,x):

        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_2(x))
        x = self.relu(self.linear_3(x))
        x = self.relu(self.linear_4(x))

        return x

### LONG SHORT-TERM MEMORY NETWORK CLASS
class LSTM(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        self.fc = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self,x):

        x, _ = self.lstm(x)
        x = self.relu(self.fc(x[:,-1,:]))
        x = x.reshape(-1,1,9)

        return x