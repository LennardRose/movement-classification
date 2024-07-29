# ---------------------------------------------------------------------------------------------------#
# File name: dataset.py                                                                             #
# Created on: 14.12.2022                                                                            #
# ---------------------------------------------------------------------------------------------------#
# Learning of Structured Data (FHWS WS22/23) - Skeleton Data time series classification
# This file provides the dataset.
# Exact description in the functions.


import os
from scipy import signal
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from helpers import *


def calc_angles(df):
    df["R_Elbow_Angle"] = compute_angle(
        np.array(list(map(list, zip(df["R_Elbow_coord_x"], df["R_Elbow_coord_y"])))),
        np.array(list(map(list, zip(df["R_Wrist_coord_x"], df["R_Wrist_coord_y"])))),
        np.array(list(map(list, zip(df["R_Shoulder_coord_x"], df["R_Shoulder_coord_y"])))))
    df["R_Armpit_Angle"] = compute_angle(
        np.array(list(map(list, zip(df["R_Shoulder_coord_x"], df["R_Shoulder_coord_y"])))),
        np.array(list(map(list, zip(df["Neck_coord_x"], df["Neck_coord_y"])))),
        np.array(list(map(list, zip(df["R_Elbow_coord_x"], df["R_Elbow_coord_y"])))))
    df["R_Shoulder_Angle"] = compute_angle(
        np.array(list(map(list, zip(df["Neck_coord_x"], df["Neck_coord_y"])))),
        np.array(list(map(list, zip(df["R_Shoulder_coord_x"], df["R_Shoulder_coord_y"])))),
        np.array(list(map(list, zip(df["Mid_Hip_coord_x"], df["Mid_Hip_coord_y"])))))
    df["R_Hip_Angle"] = compute_angle(
        np.array(list(map(list, zip(df["Mid_Hip_coord_x"], df["Mid_Hip_coord_y"])))),
        np.array(list(map(list, zip(df["R_Hip_coord_x"], df["R_Hip_coord_y"])))),
        np.array(list(map(list, zip(df["Neck_coord_x"], df["Neck_coord_y"])))))
    df["L_Elbow_Angle"] = compute_angle(
        np.array(list(map(list, zip(df["L_Elbow_coord_x"], df["L_Elbow_coord_y"])))),
        np.array(list(map(list, zip(df["L_Wrist_coord_x"], df["L_Wrist_coord_y"])))),
        np.array(list(map(list, zip(df["L_Shoulder_coord_x"], df["L_Shoulder_coord_y"])))))
    df["L_Armpit_Angle"] = compute_angle(
        np.array(list(map(list, zip(df["L_Shoulder_coord_x"], df["L_Shoulder_coord_y"])))),
        np.array(list(map(list, zip(df["Neck_coord_x"], df["Neck_coord_y"])))),
        np.array(list(map(list, zip(df["L_Elbow_coord_x"], df["L_Elbow_coord_y"])))))
    df["L_Shoulder_Angle"] = compute_angle(
        np.array(list(map(list, zip(df["Neck_coord_x"], df["Neck_coord_y"])))),
        np.array(list(map(list, zip(df["L_Shoulder_coord_x"], df["L_Shoulder_coord_y"])))),
        np.array(list(map(list, zip(df["Mid_Hip_coord_x"], df["Mid_Hip_coord_y"])))))
    df["L_Hip_Angle"] = compute_angle(
        np.array(list(map(list, zip(df["Mid_Hip_coord_x"], df["Mid_Hip_coord_y"])))),
        np.array(list(map(list, zip(df["L_Hip_coord_x"], df["L_Hip_coord_y"])))),
        np.array(list(map(list, zip(df["Neck_coord_x"], df["Neck_coord_y"])))))
    df["Head_Angle"] = compute_angle(
        np.array(list(map(list, zip(df["Neck_coord_x"], df["Neck_coord_y"])))),
        np.array(list(map(list, zip(df["Nose_coord_x"], df["Nose_coord_y"])))),
        np.array(list(map(list, zip(df["R_Shoulder_coord_x"], df["R_Shoulder_coord_y"])))))
    df["Shoulder_Distance"] = compute_euc_distance(
        np.array(list(map(list, zip(df["R_Shoulder_coord_x"], df["R_Shoulder_coord_y"])))),
        np.array(list(map(list, zip(df["L_Shoulder_coord_x"], df["L_Shoulder_coord_y"])))))

    df["Hip_Distance"] = compute_euc_distance(
        np.array(list(map(list, zip(df["R_Hip_coord_x"], df["R_Hip_coord_y"])))),
        np.array(list(map(list, zip(df["L_Hip_coord_x"], df["L_Hip_coord_y"])))))

    df["Eye_Distance"] = compute_euc_distance(
        np.array(list(map(list, zip(df["R_Eye_coord_x"], df["R_Eye_coord_y"])))),
        np.array(list(map(list, zip(df["L_Eye_coord_x"], df["L_Eye_coord_y"])))))

    return df


def read_files_to_one(path="Portfolio3/Dataset/", doSave=True, withConfidences=True, withAngles=False,
                          onlyAngles=False, onlyUpperBody=False, onlyLowerbody=False):
    try:
        path_train = path + "train/"
        filelist_train = [csv for csv in os.listdir(path_train) if csv[-4:] == ".csv"]
        path_test = path + "test/"
        filelist_test = [csv for csv in os.listdir(path_test) if csv[-4:] == ".csv"]

    except Exception as e:
        print(e)

    # prepare column selection
    # original data column names
    column_names = get_column_names()
    # columns we wanna keep
    columns_to_keep = get_bodyparts_columns(withConfidences)

    if onlyAngles:
        columns_to_keep = get_angle_column_names(True)
    elif onlyUpperBody:
        columns_to_keep = get_upper_bodyparts_columns()
        if withAngles:
            columns_to_keep += get_angle_column_names(True)
    elif onlyLowerbody:
        columns_to_keep = get_lower_bodyparts_columns()
        if withAngles:
            columns_to_keep += get_angle_column_names(True)
    elif withAngles:
        columns_to_keep += get_angle_column_names(True)
    else:
        columns_to_keep += get_angle_column_names(False)
    # label is also kept
    columns_to_keep.append("Label")

    # init df
    dataset_train = pd.DataFrame()
    dataset_test = pd.DataFrame()
    train_ts, test_ts = [], []

    # read training data
    for csv in tqdm(filelist_train, desc="Train files"):  # file_list_train:

        label = csv.split("_")[1].split(".")[0]  # get the label from the filename

        dataset_temp = pd.read_csv(path_train + csv, names=column_names)
        dataset_temp["Label"] = encode_label(label)  # add a new column to the dataframe, and assign the label to it

        if withAngles or onlyAngles:
            dataset_temp = calc_angles(dataset_temp)

        dataset_train = pd.concat([dataset_train, dataset_temp[columns_to_keep]])

        # prepare the time series dataset
        train_ts.append(dataset_temp[columns_to_keep].to_numpy())

    # read test data
    for csv in tqdm(filelist_test, desc="Test files"):
        # get name of csv before .csv
        label = csv.split(".")[0]  # get the label from the filename

        dataset_temp = pd.read_csv(path_test + csv, names=column_names)
        dataset_temp["Label"] = int(label)  # add a new column to the dataframe, and assign the label to it

        if withAngles or onlyAngles:
            dataset_temp = calc_angles(dataset_temp)

        dataset_test = pd.concat([dataset_test, dataset_temp[columns_to_keep]])
        test_ts.append(dataset_temp[columns_to_keep].to_numpy())

    dataset_train = dataset_train.fillna(0.0)  # fill the NaN values with 0
    dataset_test = dataset_test.fillna(0.0)  # fill the NaN values with 0

    # save datasets
    if doSave:
        name = ""
        if withAngles and not onlyAngles:
            name += "_withAngles"
        if onlyAngles:
            name += "_onlyAngles"
        if onlyUpperBody:
            name += "_onlyUpperBody"
        if onlyLowerbody:
            name += "_onlyLowerbody"

        np.save(path + "train_ts" + name + ".npy", train_ts, allow_pickle=True)
        np.save(path + "test_ts" + name + ".npy", test_ts, allow_pickle=True)

        dataset_train.to_csv(path + "train" + name + ".csv", index=False)
        dataset_test.to_csv(path + "test" + name + ".csv", index=False)

    return dataset_train, dataset_test


def normalize(dataset_train, dataset_test):
    """This method scales / normalises the features.
    """

    scaler = RobustScaler()
    dataset_train_scaled = scaler.fit_transform(dataset_train)
    dataset_test_scaled = scaler.transform(dataset_test)

    return dataset_train_scaled, dataset_test_scaled, scaler


def sliding_window(dataset, labels, window_size, step_size):
    """This function executes the sliding window.
    """

    # create a list of all lists
    columns = dataset.shape[1]
    feature_list = [[] for i in range(0, columns)]

    labels_list = []

    # creating overlaping windows of size window-size
    for i in range(0, dataset.shape[0] - window_size, step_size):

        # iterate over the all features / columns
        for j in range(0, columns):
            feature_list[j].append(dataset[i: i + window_size, j])

        labels_list.append(stats.mode(labels[i: i + window_size])[0][0])

    dataset_sw = np.stack(feature_list, axis=2)

    return dataset_sw, labels_list, feature_list


def reshape_cnn(dataset_train, dataset_test):
    """This method reshapes the data for CNN models.
    """

    dataset_train = np.expand_dims(dataset_train, axis=3)
    dataset_test = np.expand_dims(dataset_test, axis=3)

    return dataset_train, dataset_test


def reshape_conv_lstm(dataset_train, dataset_test):
    """This method reshapes the data for Convolutional LSTM models.
    """

    dataset_train, dataset_test = reshape_cnn(dataset_train, dataset_test)

    shape_train = dataset_train.shape
    shape_test = dataset_test.shape
    subsequences = 4  # split the sliding window into four parts

    dataset_train = dataset_train.reshape((shape_train[0], subsequences, int(shape_train[1] / subsequences),
                                           shape_train[2], 1))

    dataset_test = dataset_test.reshape((shape_test[0], subsequences, int(shape_test[1] / subsequences),
                                         shape_test[2], 1))

    return dataset_train, dataset_test


def calculate_fft(feature_list):
    """This function calculate the Fast Fourier Transform (FFT) of the data.
    """

    fft_feature_list = []

    for j in range(0, 79):
        list_fft = pd.Series(feature_list[j]).apply(lambda x: np.abs(np.fft.fft(x)))
        fft_feature_list.append(list_fft)

    return fft_feature_list


def statistical_feature_engineering(feature_list):
    """This function calculate statistical features.
       Inspired by: https://towardsdatascience.com/feature-engineering-on-time-series-data-transforming-signal-data-of-a-smartphone-accelerometer-for-72cbe34b8a60
                    Chrissi2802 https://github.com/Chrissi2802

    Args:
        feature_list (list): Dataset

    Returns:
        df (pandas DataFrame): Extended dataset
    """

    df = pd.DataFrame()

    for j in range(0, 79):
        column = str(j)
        # Feature engineering for data
        df[column + "_mean"] = pd.Series(feature_list[j]).apply(lambda x: np.mean(np.float64(x)))  # mean
        df[column + "_std"] = pd.Series(feature_list[j]).apply(lambda x: np.std(np.float64(x)))  # std dev
        df[column + "_mad"] = pd.Series(feature_list[j]).apply(
            lambda x: np.mean(np.absolute(np.float64(x) - np.mean(np.float64(x)))))  # mean absolute difference
        df[column + "_max"] = pd.Series(feature_list[j]).apply(lambda x: np.max(np.float64(x)))  # maximum
        df[column + "_min"] = pd.Series(feature_list[j]).apply(lambda x: np.min(np.float64(x)))  # minimum
        df[column + "_max_min_diff"] = df[column + "_max"] - df[column + "_min"]  # max-min difference, range
        df[column + "_median"] = pd.Series(feature_list[j]).apply(lambda x: np.median(np.float64(x)))  # median
        df[column + "_mad"] = pd.Series(feature_list[j]).apply(
            lambda x: np.median(np.absolute(np.float64(x) - np.median(np.float64(x)))))  # median absolute difference
        df[column + "_iqr"] = pd.Series(feature_list[j]).apply(
            lambda x: np.percentile(np.float64(x), 75) - np.percentile(np.float64(x), 25))  # interquartile range
        df[column + "_pos_count"] = pd.Series(feature_list[j]).apply(
            lambda x: np.sum(np.float64(x) >= 0.0))  # positive count
        df[column + "_neg_count"] = pd.Series(feature_list[j]).apply(
            lambda x: np.sum(np.float64(x) < 0.0))  # negative count
        df[column + "_tot_count"] = df[column + "_pos_count"] + df[column + "_neg_count"]  # total count
        df[column + "_above_mean"] = pd.Series(feature_list[j]).apply(
            lambda x: np.sum(np.float64(x) > np.mean(np.float64(x))))  # values above mean
        df[column + "_peak_count"] = pd.Series(feature_list[j]).apply(
            lambda x: len(signal.find_peaks(np.float64(x))[0]))  # number of peaks
        df[column + "_skewness"] = pd.Series(feature_list[j]).apply(lambda x: stats.skew(np.float64(x)))  # skewness
        df[column + "_kurtosis"] = pd.Series(feature_list[j]).apply(lambda x: stats.kurtosis(np.float64(x)))  # kurtosis
        df[column + "_energy"] = pd.Series(feature_list[j]).apply(
            lambda x: np.sum(np.float64(x) ** 2) / 100.0)  # energy
        df[column + "_sma"] = pd.Series(feature_list[j]).apply(
            lambda x: np.sum(np.absolute(np.float64(x)) / 100.0))  # signal magnitude area

        # Feature engineering for indices
        df[column + "_argmax"] = pd.Series(feature_list[j]).apply(
            lambda x: np.argmax(np.float64(x)))  # index of max value
        df[column + "_argmin"] = pd.Series(feature_list[j]).apply(
            lambda x: np.argmin(np.float64(x)))  # index of min value
        df[column + "_arg_diff"] = np.absolute(
            df[column + "_argmax"] - df[column + "_argmin"])  # absolute difference between above indices

        df = df.fillna(0.0)

    return df


if __name__ == "__main__":
    pass
    # read_files_to_one_csv("data/", doSave=True, withAngles=True)
    # read_files_to_one_csv("data/", doSave=True, withAngles=False)