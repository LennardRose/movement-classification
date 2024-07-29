import os
import pandas as pd
import numpy as np


class EnsemblePredictor:

    def __init__(self, pathToFolder):

        self.files = [pathToFolder + csv for csv in os.listdir(pathToFolder) if csv[-4:] == ".csv"]
        self.weights = []  # tbd
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.weighted_mode.html

    def _readData(self):

        data = pd.DataFrame()

        for idx, file in enumerate(self.files):
            if idx == 0:
                data = pd.read_csv(file)
                data.columns = ["id", file]
                data.set_index("id", inplace=True)

            else:
                data2 = pd.read_csv(file)
                data[file] = data2["action"]

        self.data = data

    def predict(self, doSave=True):
        self._readData()

        self.predictions = self.data.mode(axis=1)[0].to_frame()
        self.predictions.columns = ["action"]
        self.predictions["action"] = self.predictions["action"].astype(int)

        if doSave:
            self.predictions.to_csv("Results/Ensemble/submission_ensemble.csv", index_label="id", index=True,
                                    header=True)
        return self.predictions


if __name__ == "__main__":
    pred = EnsemblePredictor("ensembleSubmissions/")
    pred.predict()
