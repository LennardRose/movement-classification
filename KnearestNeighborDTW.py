import numpy as np
import pandas as pd
import tslearn
import timeit
import os

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset
from sklearn.model_selection import train_test_split
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


# TODO in andere files auslagern
def train_knn(normalize=True):
    os.makedirs("models", exist_ok=True)

    # Load data
    train = np.load("data/train_ts_onlyAngles.npy", allow_pickle=True)
    test = np.load("data/test_ts_onlyAngles.npy", allow_pickle=True)

    # Convert to X, y
    ytrain = []
    Xtrain = []
    for dataset in train:
        ytrain.append(dataset[:, -1][0])
        Xtrain.append(dataset[:, :-1])
    ytrain = np.array(ytrain, dtype=int)
    Xtrain = np.array(Xtrain, dtype=object)

    fileId = []
    X_test = []
    for dataset in test:
        fileId.append(dataset[:, -1][0])
        X_test.append(dataset[:, :-1])
    fileId = np.array(fileId, dtype=int)
    X_test = np.array(X_test, dtype=object)

    X_test_ts = to_time_series_dataset(X_test)
    X_test_ts = np.nan_to_num(X_test_ts)
    if normalize:
        X_test_ts = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_test_ts)

    Xtrain_ts = to_time_series_dataset(Xtrain)
    Xtrain_ts = np.nan_to_num(Xtrain_ts)
    if normalize:
        Xtrain_ts = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(Xtrain_ts)

    X_train_ts, X_val, y_train, y_val = train_test_split(Xtrain_ts, ytrain, test_size=0.2, random_state=42,
                                                         shuffle=False)

    start = timeit.default_timer()
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=5, metric="dtw", verbose=51, n_jobs=-1)
    clf.fit(X_train_ts, y_train)
    score = clf.score(X_val, y_val)
    stop = timeit.default_timer()
    print('Training Time: ', (stop - start) / 60)
    print('Score:', score)

    if normalize:
        clf.to_pickle("models/knn_dtw_scaled_{:.2f}acc.pkl".format(score))
    else:
        clf.to_pickle("models/knn_dtw_{:.2f}acc.pkl".format(score))
    print("Model saved...")

    start = timeit.default_timer()
    preds = clf.predict(X_test_ts)
    stop = timeit.default_timer()
    print('Predicting Time: ', (stop - start) / 60)

    sortIndices = np.argsort(fileId)
    preds = preds[sortIndices]
    if normalize:
        fileName = "Results/KNN_DTW/submission_KNN_DTW_normalizedData_{:.2f}acc.csv".format(score)
    else:
        fileName = "Results/KNN_DTW/submission_KNN_DTW_{:.2f}acc.csv".format(score)
    pd.DataFrame(np.vstack((np.arange(len(preds)), preds)).T, columns=["id", "action"]).to_csv(
        "Results/KNN_DTW/submission_KNN_DTW_normalizedData_80acc.csv", index=False)
    print("Predictions saved...")


if __name__ == "__main__":
    train_knn(normalize=False)
