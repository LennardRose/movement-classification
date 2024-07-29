import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import stats
from tqdm import tqdm

# pytorch implementation of sliding window dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, data, labels=None, window_size=200, step_size=100, moving_average=False, moving_window_size=10):
        X = []
        y = []

        for i in range(0, len(data) - window_size, step_size):

            if moving_average:
                raise NotImplementedError("Moving average not implemented yet")
                x_averaged = data['x'].iloc[i: i + window_size].rolling(moving_window_size, min_periods=1).mean()
                y_averaged = data['y'].iloc[i: i + window_size].rolling(moving_window_size, min_periods=1).mean()
                z_averaged = data['z'].iloc[i: i + window_size].rolling(moving_window_size, min_periods=1).mean()

                X.append([x, y, z, x_averaged, y_averaged, z_averaged])
            else:

                X.append(data[i: i + window_size])

            # Label for a data window is most frequent label in window
            label = stats.mode(labels[i: i + window_size], keepdims=False).mode
            y.append(label)

        X = np.array(X)#.transpose(0, 2, 1)
        y = np.array(y)

        self.y = torch.tensor(y).long()
        self.X = torch.tensor(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class SlidingWindowDatasetCNN(Dataset):
    def __init__(self, data, labels=None, window_size=200, step_size=100):
        X = []
        y = []
        
        for i in tqdm(range(0, len(data) - window_size, step_size)):
            X.append(data[i: i + window_size])
        
            # Label for a data window is most frequent label in window
            label = stats.mode(labels[i: i + window_size], keepdims=False).mode
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        self.y = torch.tensor(y).long()
        self.X = torch.transpose(torch.tensor(X), 2, 1).float()

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]