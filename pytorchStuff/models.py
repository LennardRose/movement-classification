import torch
import torch.nn as nn
import torch.nn.functional as F

n_features = 12
n_classes = 5


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.linear_1 = nn.Linear(in_features=hidden_size, out_features=100)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=100, out_features=n_classes)

    def forward(self, x):
        out, hidden_ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.linear_2(out)

        return out


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.rnn_layers = num_layers
        self.rnn = torch.nn.RNN(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                nonlinearity="relu",  # tanh/relu
                                batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, hidden_ = self.rnn(x)
        out = self.linear(out[:, -1, :])

        return out


class CNN1D(torch.nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=5, padding=2)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=800, out_features=400)
        self.dropout = torch.nn.Dropout(p=0.8)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=400, out_features=n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
