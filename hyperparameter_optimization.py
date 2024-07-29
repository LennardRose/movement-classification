# ---------------------------------------------------------------------------------------------------#
# File name: hyperparameter_optimization.py                                                          #
# Created on: 28.12.2022                                                                             #
# ---------------------------------------------------------------------------------------------------#
# https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
# tensorboard --logdir logs/hparam_tuning


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp

from helpers import *
from dataset import *
from models import *

tf.random.set_seed(28)
np.random.seed(28)


# TODO in ml.py / models.py

def train_hp(train, targets, test, test_id_list, hparams):
    """This function performs the training and validation for hyperparameter optimization."""

    # Hyperparameter
    epochs = 500  # For testing 2
    batch_size = hparams[HP_BATCH_SIZE]
    verbose = 1

    # Hardware config
    strategy = hardware_config("GPU")

    # Disable AutoShard
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    with strategy.scope():
        path_models = "Models_hp/"

        # Train / validation split
        x_train, x_valid, y_train, y_valid = train_test_split(train, targets, test_size=0.2, random_state=28)
        val_acc_last, epochs_last = [], []

        # Wrap data in Dataset objects
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).with_options(options)
        valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size).with_options(options)

        # Model, choose one
        model = mlp_net(train)
        # model = cnn_net(train)
        # model = gru_net(train)
        # model = conv_lstm_net(train)
        # model = resnet18(train)
        # model = resnet34(train)
        print(model.summary())

        model.compile(optimizer=hparams[HP_OPTIMIZER], loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        learning_rate = ReduceLROnPlateau(monitor="val_loss", factor=hparams[HP_LR_FACTOR],
                                          patience=hparams[HP_LR_PATIENCE],
                                          verbose=verbose)
        early_stopping = EarlyStopping(monitor="val_loss", patience=hparams[HP_ES_PATIENCE], verbose=verbose,
                                       mode="min",
                                       restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(path_models + model.name + ".hdf5", monitor="val_loss",
                                           verbose=verbose, save_best_only=True, mode="auto", save_freq="epoch")

        # Training
        history = model.fit(train_data,
                            validation_data=valid_data,
                            epochs=epochs,
                            verbose=2,  # for debugging verbose
                            batch_size=batch_size,
                            callbacks=[learning_rate, early_stopping, model_checkpoint])

        val_acc_last.append(np.around(100.0 * history.history["val_accuracy"][-1],
                                      2))  # safe last validation accuracy
        epochs_last.append(len(history.history["loss"]))  # safe last epoch

        print()
        print("Last validation accuracy and last epoch for each fold:", val_acc_last, epochs_last)
        print("Training, validation and testing completed!")

        return history.history["val_accuracy"][-1]


if __name__ == "__main__":

    Pr = ProgramRuntime()

    try:
        session_num = 0

        df_train, df_test = read_files_to_one(doSave=False, withAngles=False)

        df_train_data, labels = split_data_labels(df_train)
        df_test_data, labels_test = split_data_labels(df_test)

        labels_encoded = encode_labels(labels)

        df_train_data_scaled, df_test_data_scaled, scaler = normalize(df_train_data, df_test_data)

        dataset_sw, labels_list, _ = sliding_window(df_train_data_scaled, labels_encoded, 100, 50)
        dataset_sw_test, labels_list_test, _ = sliding_window(df_test_data_scaled, labels_test, 100, 50)

        # comment in for cnn_net
        # dataset_sw, dataset_sw_test = reshape_cnn(dataset_sw, dataset_sw_test)

        # comment in for conv_lstm_net
        # dataset_sw, dataset_sw_test = reshape_conv_lstm(dataset_sw, dataset_sw_test)

        HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 128, 256]))
        HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
        HP_LR_FACTOR = hp.HParam('lr_factor', hp.Discrete([0.1, 0.5, 0.7]))
        HP_LR_PATIENCE = hp.HParam('lr_patience', hp.Discrete([5, 10, 15]))
        HP_ES_PATIENCE = hp.HParam('es_patience', hp.Discrete([10, 50, 70]))

        # Hyperparameter tuning
        for batch_size in HP_BATCH_SIZE.domain.values:
            for optimizer in HP_OPTIMIZER.domain.values:
                for lr_factor in HP_LR_FACTOR.domain.values:
                    for lr_patience in HP_LR_PATIENCE.domain.values:
                        for es_patience in HP_ES_PATIENCE.domain.values:
                            hparams = {
                                HP_BATCH_SIZE: batch_size,
                                HP_OPTIMIZER: optimizer,
                                HP_LR_FACTOR: lr_factor,
                                HP_LR_PATIENCE: lr_patience,
                                HP_ES_PATIENCE: es_patience,
                            }

                            run_name = "run-%d" % session_num
                            print('--- Starting trial: %s' % run_name)
                            print({h.name: hparams[h] for h in hparams})

                            # run
                            run_dir = 'logs/hparam_tuning/' + run_name
                            with tf.summary.create_file_writer(run_dir).as_default():
                                hp.hparams(hparams)  # record the values used in this trial
                                accuracy = train_hp(dataset_sw, np.array(labels_list), dataset_sw_test,
                                                    np.array(labels_list_test), hparams)
                                tf.summary.scalar("val_accuracy", accuracy, step=1)

                            session_num += 1

    except Exception as e:
        print(e)
    finally:
        runtime = Pr.finish(print=True)

    # Best results with hyperparameter optimization
# mlp_net
# lr_factor 0,5
# es_patience 50
# lr_patience 10
# optimizer Adam
# batch_size 256

# gru_net
# lr_patience 15
# es_patience 50
# batch_size 256
# lr_factor 0,7
# optimizer Adam

# cnn_net
# optimizer Adam
# lr_patience 5
# batch_size 64
# es_patience 70
# lr_factor 0,5
