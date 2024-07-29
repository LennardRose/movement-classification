import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from SlidingWindow import SlidingWindowDataset
from models import LSTM


def split_data_labels(dataset_train):
    labels = dataset_train["Label"]
    dataset_train = dataset_train.drop("Label", axis=1)

    return dataset_train, labels


def connect_to_gpu():
    if torch.cuda.is_available():
        DEVICE = "cuda"
        # Clear cache if non-empty
        torch.cuda.empty_cache()
        # See which GPU has been allotted
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        DEVICE = "cpu"
        print("On CPU")

    return DEVICE


def doStuff():
    df_train_set = pd.read_csv("../data/train_withAngles.csv", )
    df_test_set = pd.read_csv("../data/test_withAngles.csv")
    df_train_data, labels = split_data_labels(df_train_set)
    df_test_data, file_id = split_data_labels(df_test_set)

    train_data_scaled = StandardScaler().fit_transform(df_train_data)
    test_data_scaled = StandardScaler().fit_transform(df_test_data)

    X_train, X_val, y_train, y_val = train_test_split(train_data_scaled, labels, test_size=0.3, random_state=42,
                                                      shuffle=False)

    train_set = SlidingWindowDataset(X_train, y_train, window_size=100, step_size=50)
    val_set = SlidingWindowDataset(X_val, y_val, window_size=100, step_size=50)
    test_set = SlidingWindowDataset(test_data_scaled, file_id, window_size=100, step_size=50)

    # Connect to GPU
    DEVICE = connect_to_gpu()

    lstm = LSTM(input_size=62, hidden_size=128, num_layers=1)

    lstm = lstm.to(DEVICE)

    lstm, (train_acc, val_acc), = train_model(lstm, epochs=200, batch_size=128, learning_rate=0.0025,
                                              train_set=train_set,
                                              val_set=val_set, device=DEVICE, disable_progress=True)

    torch.save(lstm.state_dict(), "../models/lstm_model_{val_acc:.0f}.pt".format(val_acc=val_acc * 100))

    predict_test(DEVICE, lstm, test_set, val_acc)


def predict_test(DEVICE, lstm, test_set, val_acc):
    dl_test = DataLoader(test_set, 1, shuffle=False)
    classifications = []
    for inputs, file_id in dl_test:
        torch.cuda.empty_cache()
        inputs = inputs.to(DEVICE)
        classifications.append([file_id.item(), torch.argmax(lstm(inputs), 1).item()])
    fileName = "../Results/LSTMtorch/submission_lstm_pytorch_val{val_acc:.0f}.csv".format(val_acc=val_acc * 100)
    pd.DataFrame(classifications, columns=["id", "action"]).groupby("id").agg(
        lambda x: x.value_counts().index[0]).reset_index().to_csv(fileName, index=False, sep=",")


def doStuffWithKfold():
    df_train_set = pd.read_csv("../data/train_withAngles.csv", )
    df_test_set = pd.read_csv("../data/test_withAngles.csv")
    df_train_data, labels = split_data_labels(df_train_set)
    df_test_data, file_id = split_data_labels(df_test_set)

    train_data_scaled = StandardScaler().fit_transform(df_train_data)
    test_data_scaled = StandardScaler().fit_transform(df_test_data)

    #X_train, X_val, y_train, y_val = train_test_split(train_data_scaled, labels, test_size=0.0, random_state=42,
    #                                                  shuffle=False)

    train_set = SlidingWindowDataset(train_data_scaled, labels, window_size=100, step_size=50)
    #val_set = SlidingWindowDataset(X_val, y_val, window_size=100, step_size=50)
    test_set = SlidingWindowDataset(test_data_scaled, file_id, window_size=100, step_size=50)

    # Connect to GPU
    DEVICE = connect_to_gpu()

    lstm, (train_acc, val_acc), = train_model_kfold(epochs=100, batch_size=128, learning_rate=0.0025,
                                              dataset=train_set, device=DEVICE, disable_progress=True)

    torch.save(lstm.state_dict(), "../models/lstm_model_{val_acc:.0f}.pt".format(val_acc=val_acc * 100))

    predict_test(DEVICE, lstm, test_set, val_acc)


def train_model_kfold(epochs, batch_size, learning_rate, dataset, doPlots=True, disable_progress=False,
                      device='cpu'):
    """Method to train a lstm with given parameters.

    Parameters
    ----------
    model : pytorch.Model
        Instantiated pytorch model.
    epochs : int
        Amount of epochs to train.
    batch_size : int
        Size of the batches while training.
    learning_rate : float
        Determines how big the update steps should be.
    train_set : dataset
        Training part of the dataset.
    val_set : dataset
        Validation part of the dataset.
    doPlots : bool, optional
        Boolean that defines if training/testing plots should be shown after finishing training.
    disable_progress : bool, optional
        Boolean that defines if progress bars should be shown while training.
    device : str, optional
        Device to train on.

    Returns
    -------
    pytorch.Model, tuple(list(float), list(float))
        Returns both the trained model and the training and testing losses.
    """

    # history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    k = 5
    splits = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold + 1))

        # Create samplers
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)

        # Create data loaders
        dl_train = DataLoader(dataset, batch_size, shuffle=False, sampler=train_sampler)
        dl_test = DataLoader(dataset, batch_size, shuffle=False, sampler=test_sampler)

        model = LSTM(input_size=62, hidden_size=128, num_layers=1)
        model = model.to(device)

        # Define loss function and optimizer
        # optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)
        optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01, last_epoch=-1, verbose=False)
        scheduler = None
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            if epoch % (epochs * 0.05) == 0:
                print(f"Epoch: {epoch + 1:02}/{epochs:02}")

            train_loss, train_acc = train_epoch(model, dl_train, optimizer, scheduler, criterion, disable_progress,
                                                device)
            val_loss, val_acc = validate_epoch(model, dl_test, criterion, disable_progress, device)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if epoch % (epochs * 0.05) == 0:
                print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")
                print(f"Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc * 100:.2f}%")

    if doPlots:
        plt.plot(range(epochs*k), train_losses)
        plt.plot(range(epochs*k), val_losses)
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend(["Train", "Val."])
        plt.show()

        plt.plot(range(epochs*k), train_accs)
        plt.plot(range(epochs*k), val_accs)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend(["Train", "Val."])
        plt.show()

    return model, (train_acc, val_acc)


def train_model(model, epochs, batch_size, learning_rate, train_set, val_set, doPlots=True, disable_progress=False,
                device='cpu'):
    """Method to train a lstm with given parameters.
    Parameters
    ----------
    model : pytorch.Model
        Instantiated pytorch model.
    epochs : int
        Amount of epochs to train.
    batch_size : int
        Size of the batches while training.
    learning_rate : float
        Determines how big the update steps should be.
    train_set : dataset
        Training part of the dataset.
    val_set : dataset
        Testing part of the dataset.
    doPlots : bool, optional
        Boolean that defines if training/testing plots should be shown after finishing training.
    disable_progress : bool, optional
        Boolean that defines if progress bars should be shown while training.
    device : str, optional
        Device to train on.

    Returns
    -------
    pytorch.Model, tuple(list(float), list(float))
        Returns both the trained model and the training and testing losses.
    """
    dl_train = DataLoader(train_set, batch_size, shuffle=False)
    dl_test = DataLoader(val_set, batch_size, shuffle=False)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01, last_epoch=-1, verbose=False)
    scheduler = None
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        if epoch % (epochs * 0.05) == 0:
            print(f"Epoch: {epoch + 1:02}/{epochs:02}")

        train_loss, train_acc = train_epoch(model, dl_train, optimizer, scheduler, criterion, disable_progress, device)
        val_loss, val_acc = validate_epoch(model, dl_test, criterion, disable_progress, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if epoch % (epochs * 0.05) == 0:
            print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")
            print(f"Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc * 100:.2f}%")

    if doPlots:
        plt.plot(range(epochs), train_losses)
        plt.plot(range(epochs), val_losses)
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend(["Train", "Val."])
        plt.show()

        plt.plot(range(epochs), train_accs)
        plt.plot(range(epochs), val_accs)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend(["Train", "Val."])
        plt.show()

    return model, (train_loss, val_loss)


def train_epoch(model, loader, optimizer, scheduler, criterion, disable_progress, device):
    """Method to perform one round of training the model.

    Parameters
    ----------
    model : pytorch.model
        Trained Pytorch model.
    loader : pytorch.loader
        Dataloader for loading the dataset.
    optimizer : pytorch.optimizer
        Optimizer that performs updates to the model.
    criterion : pytorch.criterion
        Criterion for loss calculation.
    disable_progress : bool
        Boolean that defines if progress bars should be shown while training.

    Returns
    -------
    epoch_loss
        Returns the loss after one epoch of training.
    """

    epoch_loss = 0
    y_pred = []
    y_true = []

    model.train()
    for batch, labels in tqdm(loader, disable=disable_progress):
        torch.cuda.empty_cache()
        batch = batch.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()

        output = model(batch)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        preds = output.argmax(dim=1)
        y_pred += preds.tolist()
        y_true += labels.tolist()
        epoch_loss += loss.item()

    if scheduler is not None:
        scheduler.step()

    # Loss
    avg_epoch_loss = epoch_loss / len(loader)

    # Accuracy
    epoch_acc = accuracy_score(y_true, y_pred)

    return avg_epoch_loss, epoch_acc


def validate_epoch(model, loader, criterion, disable_progress, device):
    """Method to perform one round of testing the model.

    Parameters
    ----------
    model : pytorch.model
        Trained Pytorch model.
    loader : pytorch.loader
        Dataloader for loading the dataset.
    criterion : pytorch.criterion
        Criterion for loss calculation.
    disable_progress : bool
        Boolean that defines if progress bars should be shown while training.

    Returns
    -------
    epoch_loss
        Returns the loss after one epoch of testing.
    """

    epoch_loss = 0
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch, labels in tqdm(loader, disable=disable_progress):
            torch.cuda.empty_cache()
            batch = batch.to(device)
            labels = labels.long().to(device)

            output = model(batch)
            loss = criterion(output, labels)

            preds = output.argmax(dim=1)
            y_pred += preds.tolist()
            y_true += labels.tolist()
            epoch_loss += loss.item()

    # Loss
    avg_epoch_loss = epoch_loss / len(loader)

    # Accuracy
    epoch_acc = accuracy_score(y_true, y_pred)

    return avg_epoch_loss, epoch_acc


def apply(model, data, batch_size):
    """Method to make predictions with a given model and test samples.

    Parameters
    ----------
    model : pytorch.model
        Trained Pytorch model.
    test_indices : list[int]
        List with some inidices for which the model should predict labels.
    """
    dl = DataLoader(data, batch_size, shuffle=False)

    output = torch.tensor([])
    model.eval()

    for X, _ in dl:
        y_hat = model(X)
        output = torch.cat((output, y_hat), 0)

    return output


if __name__ == "__main__":
    doStuffWithKfold()
