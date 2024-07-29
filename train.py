#---------------------------------------------------------------------------------------------------#
# File name: train.py                                                                               #
# Created on: 14.12.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# Learning of Structured Data (FHWS WS22/23) - Skeleton Data time series classification
# Exact description in the functions.


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from helpers import *
from dataset import *
from models import *


tf.random.set_seed(28)
np.random.seed(28)


def train(train, targets, test, test_id_list):
    """This function performs the training and testing."""

    # Hyperparameter
    epochs = 2  #500    # For testing 2
    batch_size = 256
    verbose = 1
    
    # Hardware config
    strategy = hardware_config("GPU")

    # Disable AutoShard
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    with strategy.scope():

        path_models = "Models/"

        # Crossvalidation
        k_fold = KFold(n_splits = 5, shuffle = True, random_state = 28)    # For testing n_splits = 2
        val_acc_last, epochs_last = [], []

        # Numpy array for the predictions
        test_predictions = np.empty([test.shape[0], 5, k_fold.n_splits])

        # Perform the crossvalidation
        for fold, (train_index, test_index) in enumerate(k_fold.split(train, targets)):

            print("Fold:", fold)

            # Data for this fold
            x_train, x_valid = train[train_index], train[test_index]
            y_train, y_valid = targets[train_index], targets[test_index]

            # Wrap data in Dataset objects
            train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).with_options(options)
            valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size).with_options(options)
            test_data = tf.data.Dataset.from_tensor_slices((test)).batch(batch_size).with_options(options)

            # Model, choose one
            model = mlp_net(train)
            #model = cnn_net(train)
            #model = cnn_net_v2(train)
            #model = gru_net(train)
            #model = conv_lstm_net(train)
            #model = resnet18(train)
            #model = resnet34(train)
            print(model.summary())

            model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

            learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 10, verbose = verbose)
            early_stopping = EarlyStopping(monitor = "val_loss", patience = 50, verbose = verbose, mode = "min", 
                                           restore_best_weights = True)
            model_checkpoint = ModelCheckpoint(path_models + model.name + str(fold) + ".hdf5", monitor = "val_loss", 
                                               verbose = verbose, save_best_only = True, mode = "auto", save_freq = "epoch")

            # Training
            history = model.fit(train_data, 
                                validation_data = valid_data, 
                                epochs = epochs,
                                verbose = 2,    # for debugging verbose
                                batch_size = batch_size, 
                                callbacks = [learning_rate, early_stopping, model_checkpoint])

            # Plot training and testing curves
            plot_loss_and_acc(len(history.history["loss"]), history.history["loss"],
                                      [acc * 100.0 for acc in history.history["accuracy"]], 
                                      history.history["val_loss"], 
                                      [acc * 100.0 for acc in history.history["val_accuracy"]], str(fold))
            
            # Plot confusion matrix
            valid_predictions = np.argmax(model.predict(valid_data, batch_size = batch_size), axis = 1)
            plot_conf_matrix(valid_predictions, y_valid, fold)  

            val_acc_last.append(np.around(100.0 * history.history["val_accuracy"][-1], 2))    # safe last validation accuracy                                        
            epochs_last.append(len(history.history["loss"]))    # safe last epoch

            # Save predictions 
            test_predictions[:, :, fold] = model.predict(test_data, batch_size = batch_size)

            print()

        # Save submissions
        write_submissions_max(test_predictions, test_id_list)
        print("Last validation accuracy and last epoch for each fold:", val_acc_last, epochs_last)
        print("Training, validation and testing completed!")


def train_entire_data(train, targets, test, test_id_list):
    """This function performs the training for the entire data."""

    # Hyperparameter
    epochs = 500    # For testing 2
    batch_size = 64
    verbose = 1
    
    # Hardware config
    strategy = hardware_config("GPU")

    # Disable AutoShard
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    with strategy.scope():

        path_models = "Models/"
        train, targets = shuffle(train, targets, random_state = 28)
        val_acc_last, epochs_last = [], []

        # Numpy array for the predictions
        test_predictions = np.empty([test.shape[0], 5])

        # Wrap data in Dataset objects
        train_data = tf.data.Dataset.from_tensor_slices((train, targets)).batch(batch_size).with_options(options)
        test_data = tf.data.Dataset.from_tensor_slices((test)).batch(batch_size).with_options(options)

        # Model, choose one
        #model = mlp_net(train)
        model = cnn_net(train)
        #model = cnn_net_v2(train)
        #model = gru_net(train)
        #model = conv_lstm_net(train)
        #model = resnet18(train)
        #model = resnet34(train)
        print(model.summary())

        model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

        learning_rate = ReduceLROnPlateau(monitor = "loss", factor = 0.5, patience = 10, verbose = verbose)
        early_stopping = EarlyStopping(monitor = "loss", patience = 50, verbose = verbose, mode = "min", 
                                        restore_best_weights = True)
        model_checkpoint = ModelCheckpoint(path_models + model.name + ".hdf5", monitor = "loss", 
                                            verbose = verbose, save_best_only = True, mode = "auto", save_freq = "epoch")                                

        # Training
        history = model.fit(train_data, 
                            epochs = epochs,
                            verbose = 2,    # for debugging verbose
                            batch_size = batch_size, 
                            callbacks = [learning_rate, early_stopping, model_checkpoint])

        # Plot training curves
        plot_loss_and_acc(len(history.history["loss"]), history.history["loss"],
                                    [acc * 100.0 for acc in history.history["accuracy"]], 
                                    [], 
                                    [])
         
        val_acc_last.append(np.around(100.0 * history.history["accuracy"][-1], 2))    # safe last accuracy                                    
        epochs_last.append(len(history.history["loss"]))    # safe last epoch

        # Save predictions 
        test_predictions = model.predict(test_data, batch_size = batch_size)

        predictions = pd.DataFrame(test_predictions)
        predictions["id"] = test_id_list
        predictions["id"] = predictions["id"].astype(int)

        predictions = predictions.groupby(["id"]).sum()     # Summing up classes with the same id
        predictions = predictions.sort_index()
        predictions = predictions.idxmax(axis = 1)          # find highest value and return index
        predictions = predictions.astype(int)

        submission = pd.DataFrame({"id": predictions.index, "action": predictions.values})

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        submission.to_csv(now + "_submission.csv", index = False)

        print()
        print("Last accuracy and last epoch:", val_acc_last, epochs_last)
        print("Training and testing completed!")


def classifier(x_train, x_test, y_train):

    # Choose one
    clf = LinearSVC()
    #clf.fit(x_train, y_train)
    #y_pred = clf.predict(x_test)

    lr = LogisticRegression(random_state = 28)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)

    return y_pred


if __name__ == "__main__":

    Pr = ProgramRuntime()

    try:

        df_train, df_test = read_files_to_one(doSave=False, withAngles=True, onlyAngles=False)

        # angle features
        #df_train = compute_angles(df_train)
        #df_test = compute_angles(df_test)
        
        # comment in if you want to use only the angles and labels
        #df_train = df_train.iloc[:, -13:]
        #df_test = df_test.iloc[:, -13:]

        df_train_data, labels = split_data_labels(df_train)
        df_test_data, labels_test = split_data_labels(df_test)

        labels_encoded = encode_labels(labels)

        df_train_data_scaled, df_test_data_scaled, scaler = normalize(df_train_data, df_test_data)

        dataset_sw, labels_list, feature_list = sliding_window(df_train_data_scaled, labels_encoded, 100, 50)
        dataset_sw_test, labels_list_test, feature_list_test = sliding_window(df_test_data_scaled, labels_test, 100, 50)

        # comment in for cnn_net
        #dataset_sw, dataset_sw_test = reshape_cnn(dataset_sw, dataset_sw_test)

        # comment in for conv_lstm_net
        #dataset_sw, dataset_sw_test = reshape_conv_lstm(dataset_sw, dataset_sw_test)

        #print(dataset_sw.shape, len(labels_list))
        #print(dataset_sw_test.shape, len(labels_list_test))

        train(dataset_sw, np.array(labels_list), dataset_sw_test, np.array(labels_list_test))
        #train_entire_data(dataset_sw, np.array(labels_list), dataset_sw_test, np.array(labels_list_test))

        
        # Comment in for machine learning models
        """
        # FFT
        feature_list_fft = calculate_fft(feature_list)
        feature_list_fft_test = calculate_fft(feature_list_test)

        # Statistical feature engineering on time data
        dataset_sw = statistical_feature_engineering(feature_list)
        dataset_sw_test = statistical_feature_engineering(feature_list_test)

        # Statistical feature engineering on frequency data
        dataset_sw_fft = statistical_feature_engineering(feature_list_fft)
        dataset_sw_fft_test = statistical_feature_engineering(feature_list_fft_test)

        # Combine time and frequency data
        dataset_sw = pd.concat([dataset_sw, dataset_sw_fft], axis = 1)
        dataset_sw_test = pd.concat([dataset_sw_test, dataset_sw_fft_test], axis = 1)

        scaler = StandardScaler()
        dataset_sw = scaler.fit_transform(dataset_sw)
        dataset_sw_test = scaler.transform(dataset_sw_test)

        # Classification 
        y_pred = classifier(dataset_sw, dataset_sw_test, np.array(labels_list))    
        write_submissions_ml(y_pred, np.array(labels_list_test))
        """
        
    except Exception as e:
        print(e)
    finally:
        runtime = Pr.finish(print = True)    
    
