# ---------------------------------------------------------------------------------------------------#
# File name: helpers.py                                                                             #
# Autor: Chrissi2802 https://github.com/Chrissi2802                                                 #
# Created on: 05.08.2022                                                                            #
# ---------------------------------------------------------------------------------------------------#
# Exact description in the functions.
# This file provides auxiliary classes and functions for neural networks.

import glob
import math
import re
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy import stats
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


def plot_pose(pose, bodyparts, plot=False, save=False, title=None, tolerance=5):
    # adds the edges to the graph by a fixed pattern based on the body_25 format
    def add_edges_to_graph(G):
        edges = [(0, 1), (0, 15), (0, 16),  # Nose
                 (15, 17), (16, 18),  # Eyes
                 (1, 2), (1, 5), (1, 8),  # Chest
                 (2, 3), (3, 4),  # Shoulders + Arms R
                 (5, 6), (6, 7),  # Shoulders + Arms L
                 (8, 9), (8, 12),  # Hip
                 (9, 10), (10, 11), (11, 24), (11, 22), (22, 23),  # Leg Foot R
                 (12, 13), (13, 14), (14, 21), (14, 19), (19, 20)  # Leg Foot L
                 ]

        for i in range(len(edges)):
            G.add_edge(edges[i][0], edges[i][1])

    def add_nodes_from_pose_to_graph(G, row):
        nodes = []
        for i in range(len(row) - 5):  # minus 5 because we dont need the angle attributes and the label
            node = {}
            if i % 3 == 0:
                node["position"] = (pose.iat[i], pose.iat[i + 1])  # x, y position
                node["confidence"] = round(pose.iat[i + 2], 4)
                node["bodypart"] = bodyparts[int(i / 3)]
                nodes.append((int(i / 3), node))  # add tuple of index postion and dict to fit G.add_nodes_from format

        G.add_nodes_from(nodes)  # add all nodes

    def remove_zero_nodes(G):
        # remove blank nodes for poses were for example the legs werent used, they
        # cannot be distinguished by position, mostly around 0, negative values are faulty for sure
        # if for some reason there are some faulty nodes in the graph, remove first condition from the if clause
        # because confidence values were bad
        for node in list(G.nodes.items()):  # list to get a copy
            if node[1]["confidence"] != 1 and int(node[1]["position"][0]) < tolerance and int(
                    node[1]["position"][1]) < tolerance:  # no zero values in frames
                G.remove_node(node[0])

    G = nx.Graph()
    add_nodes_from_pose_to_graph(G, pose)
    add_edges_to_graph(G)
    remove_zero_nodes(G)

    # get position for plotting
    pos = nx.get_node_attributes(G, 'position')

    # get labels
    labels = {}
    for row in list(G.nodes.items()):
        labels[row[0]] = row[1]["bodypart"]
    # add axis
    fig, ax = plt.subplots()
    nx.draw(G, pos=pos, node_color='b', ax=ax, labels=labels, with_labels=True)
    plt.axis("on")
    ax.set_xlim(0, 640)  # image dimensions
    ax.set_ylim(0, 480)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.gca().invert_yaxis()

    if save:
        plt.savefig("./images/" + title + ".png")
    if plot:
        plt.show()

    # clear figure
    plt.close()


def save_as_gif(input_folder="./images/*", output_folder="./images/poses.gif"):
    """
    saves all images from the input_folder as one gif, sorts by numers in filename
    https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    :param input_folder: The directory to get all images from
    :param output_folder: The directory to save the created gif to
    """
    imgs = (Image.open(f) for f in sorted(glob.glob(input_folder), key=get_order))
    img = next(imgs)  # extract first image from iterator
    img.save(fp=output_folder, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)


def get_order(file):
    """
    For stop sorting like 1, 10, 100, 11 .. 19, 2, 20, 21 ...
    recklessly stolen from https://stackoverflow.com/questions/62941378/how-to-sort-glob-glob-numerically
    :param file: the file to be sorted
    """
    file_pattern = re.compile(r'.*?(\d+).*?')
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])


def plot_loss_and_acc(epochs, train_losses, train_acc, test_losses=None, test_acc=None, fold=-1):
    """This function plots the loss and accuracy for training and, if available, for validation."""
    # Input:
    # epochs; integer, Number of epochs
    # train_losses; list, Loss during training for each epoch
    # train_acc; list, Accuracy during training for each epoch
    # test_losses; list default [], Loss during validation for each epoch
    # test_acc; list default [], Accuracy during validation for each epoch
    # fold; integer default -1, Cross-validation run

    if test_acc is None:
        test_acc = []
    if test_losses is None:
        test_losses = []
    fig, ax1 = plt.subplots()
    xaxis = list(range(1, epochs + 1))

    # Training
    # Loss
    trl = ax1.plot(xaxis, train_losses, label="Training Loss", color="red")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    # Accuracy
    ax2 = ax1.twinx()
    tra = ax2.plot(xaxis, train_acc, label="Training Accuracy", color="fuchsia")
    ax2.set_ylabel("Accuracy in %")
    ax2.set_ylim(0.0, 100.0)
    lns = trl + tra  # Labels

    # Test
    if (test_losses != []) and (test_acc != []):
        # Loss
        tel = ax1.plot(xaxis, test_losses, label="Validation Loss", color="lime")

        # Accuracy
        tea = ax2.plot(xaxis, test_acc, label="Validation Accuracy", color="blue")

        lns = trl + tel + tra + tea  # Labels

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    if fold == -1:
        fold1 = ""
        fold2 = ""
    else:
        fold1 = " Fold " + str(fold)
        fold2 = "_Fold_" + str(fold)

    plt.title("Loss and Accuracy" + fold1)
    fig.savefig("Loss_and_Accuracy" + fold2 + ".png")
    plt.show()


def plot_conf_matrix(valid_predictions, y_valid, fold):
    """This function plots the confusion matrix for the validation data."""
    # Input:
    # valid_predictions; NumPy array, Array of predictions
    # y_valid; NumPy array, Array of the true labels
    # fold; integer, Cross-validation run

    class_names = ["boxing", "drums", "guitar", "rowing", "violin"]
    conf_matrix = confusion_matrix(y_valid, valid_predictions)
    plot_confusion_matrix(conf_mat = conf_matrix, class_names = class_names, show_normed = True, figsize = (10, 7), colorbar = True)
    plt.title("Confusion matrix Fold " + str(fold))
    plt.savefig("Confusion_matrix_Fold_" + str(fold) + ".png")   
    plt.show()


class ProgramRuntime():
    """Class for calculating the programme runtime and outputting it to the console."""

    def __init__(self):
        """Initialisation of the class (constructor). Automatically saves the start time."""

        self.begin()

    def begin(self):
        """This method saves the start time."""

        self.__start = datetime.now()  # start time

    def finish(self, print=True):
        """This method saves the end time and calculates the runtime."""
        # Input:
        # print; boolean, default false, the start time, end time and the runtime should be output to the console
        # Output:
        # self.__runtime; integer, returns the runtime

        self.__end = datetime.now()  # end time
        self.__runtime = self.__end - self.__start  # runtime

        if (print == True):
            self.show()

        return self.__runtime

    def show(self):
        """This method outputs start time, end time and the runtime on the console."""

        print()
        print("Start:", self.__start.strftime("%Y-%m-%d %H:%M:%S"))
        print("End:  ", self.__end.strftime("%Y-%m-%d %H:%M:%S"))
        print("Program runtime:", str(self.__runtime).split(".")[0])  # Cut off milliseconds
        print()


def hardware_config(device="GPU"):
    """This function configures the hardware."""
    # Input:
    # device; string default GPU, which device to use, TPU or GPU
    # Output:
    # strategy; tensorflow MirroredStrategy

    if (device == "TPU"):
        # TPU, use only if TPU is available
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        # GPU, if not available, CPU is automatically selected
        gpus = tf.config.list_logical_devices("GPU")
        strategy = tf.distribute.MirroredStrategy(gpus)

    return strategy


def split_data_labels(dataset_train):
    """This function separates the data from the labels.
    """

    labels = dataset_train["Label"]
    dataset_train = dataset_train.drop("Label", axis=1)

    return dataset_train, labels


def write_submissions_max(test_predictions, test_id_list):
    """This function writes the predicted labels to a csv file.
       The most frequent label is used from the cross-validation.
    """

    label_dic_inv = {0: "boxing", 1: "drums", 2: "guitar", 3: "rowing", 4: "violin"}

    # Every single fold is used. The folds are added, then the same ids are added and the maximum is determined.
    predictions = np.sum(test_predictions, axis=2)  # Adding the Folds

    predictions = pd.DataFrame(predictions)
    predictions["id"] = test_id_list
    predictions["id"] = predictions["id"].astype(int)

    predictions = predictions.groupby(["id"]).sum()  # Summing up classes with the same id
    predictions = predictions.sort_index()
    predictions = predictions.idxmax(axis=1)  # find highest value and return index

    submission = pd.DataFrame({"id": predictions.index, "action": predictions.values})
    # submission["action"] = submission["action"].map(label_dic_inv)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    submission.to_csv(now + "_max_submission.csv", index=False)


def write_submissions_ml(y_pred, labels_list_test):
    """This function writes the predicted labels to a CSV file.
    """

    predictions = pd.DataFrame(y_pred, columns=["Predictions"])
    predictions["id"] = labels_list_test
    predictions["id"] = predictions["id"].astype(int)

    # Grouping by id and uses only the value which is the most common
    predictions = predictions.groupby(["id"]).agg(lambda x: x.value_counts().index[0])
    predictions = predictions.sort_index()

    submission = pd.DataFrame({"id": predictions.index, "action": predictions["Predictions"]})
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    submission.to_csv(now + "_ml_submission.csv", index=False)


def encode_label(label):
    label_dic = {"boxing": 0, "drums": 1, "guitar": 2, "rowing": 3, "violin": 4}

    return label_dic[label]


def encode_labels(labels):
    """This function encodes the labels to digits.
    """

    label_dic = {"boxing": 0, "drums": 1, "guitar": 2, "rowing": 3, "violin": 4}

    # encode the labels
    labels_encoded = [label_dic[label] for label in labels]

    return labels_encoded


def get_upper_bodyparts():
    return ["Nose", "Neck", "R_Shoulder", "R_Elbow", "R_Wrist", "L_Shoulder", "L_Elbow", "L_Wrist", "R_Eye", "L_Eye",
            "R_Ear", "L_Ear", "Mid_Hip"]


def get_lower_bodyparts():
    return ["Mid_Hip", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle", "R_BigToe", "R_SmallToe", "R_Heel",
            "L_BigToe", "L_SmallToe", "L_Heel"]


def get_body_parts(upperBody=True, lowerBody=True):
    """
    Generates body parts based on the boolean values.
    Mid Hip is inluded in upperbody.
    """
    bodyparts = []
    if upperBody:
        bodyparts.extend(get_upper_bodyparts())
    if lowerBody:
        bodyparts.extend(get_lower_bodyparts())

    if upperBody and lowerBody:
        bodyparts.remove("Mid_Hip")
    return bodyparts

    # return ["Nose", "Neck", "R_Shoulder", "R_Elbow", "R_Wrist", "L_Shoulder", "L_Elbow", "L_Wrist", "Mid_Hip",
    #             "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle", "R_Eye", "L_Eye", "R_Ear", "L_Ear",
    #             "L_BigToe", "L_SmallToe", "L_Heel", "R_BigToe", "R_SmallToe", "R_Heel"]


def get_bodyparts_with_infos(bodyparts, withConfidence=False):
    columns = []
    for bodypart in bodyparts:
        columns.append(bodypart + "_coord_x")
        columns.append(bodypart + "_coord_y")
        if withConfidence:
            columns.append(bodypart + "_confidence")

    return columns


def sliding_window(dataset, labels, window_size, step_size):
    data = []
    labels_list = []

    for i in range(0, dataset.shape[0] - window_size, step_size):
        data.append(dataset[i: i + window_size])
        labels_list.append(stats.mode(labels[i: i + window_size], keepdims=False).mode)

    return np.array(data), np.array(labels_list)


def get_bodyparts_columns(withConfidence=False):
    cols = get_bodyparts_with_infos(get_body_parts(), withConfidence)
    return cols


def get_upper_bodyparts_columns(withConfidence=False):
    cols = get_bodyparts_with_infos(get_body_parts(upperBody=True, lowerBody=False), withConfidence)
    # cols.extend(["R_Elbow_Angle", "R_Armpit_Angle", "L_Elbow_Angle", "L_Armpit_Angle"])
    return cols


def get_lower_bodyparts_columns(withConfidence=False):
    return get_bodyparts_with_infos(get_body_parts(upperBody=False, lowerBody=True), withConfidence)


def get_column_names(withLabel=False):
    column_names = get_bodyparts_with_infos(get_body_parts(), withConfidence=True)

    column_names.append("R_Elbow_Angle")
    column_names.append("R_Armpit_Angle")
    column_names.append("L_Elbow_Angle")
    column_names.append("L_Armpit_Angle")

    if withLabel:
        column_names.append("Label")

    return column_names


def get_confidence_names():
    return [bodypart + "_confidence" for bodypart in get_body_parts()]


def get_angle_column_names(augmented=True):
    if augmented:
        return ["Head_Angle", "R_Elbow_Angle", "R_Armpit_Angle", "L_Armpit_Angle", "L_Elbow_Angle",
                "L_Hip_Angle", "R_Hip_Angle", "L_Shoulder_Angle", "R_Shoulder_Angle",
                "Eye_Distance", "Shoulder_Distance", "Hip_Distance"]
    else:
        return ["R_Elbow_Angle", "R_Armpit_Angle", "L_Armpit_Angle", "L_Elbow_Angle"]


def compute_euc_distance(p1, p2):
    """
    computes the euclidean distance between 2 vectors rowwise
    """
    return np.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)


def compute_angle(a, b, c):
    """
    computes the angle between each row
    NOTE a IS THE VERTEX!
    returns the angle in degrees
    """
    ab = compute_euc_distance(a, b)
    ac = compute_euc_distance(a, c)
    bc = compute_euc_distance(b, c)

    return np.rad2deg(np.arccos((ab ** 2 + ac ** 2 - bc ** 2) / (2 * ab * ac)))


def plot_total_params_accuracy(df):
    """This function plots to the trained models and the number of all parameters and the achieved accuracy on kaggle.

    Args:
        df (DataFrame): Models, total parameters, accuracy and ratio
    """

    fig, ax1 = plt.subplots()
    plt.subplots_adjust(bottom = 0.25)
    xaxis = df["Model"]

    # Total params
    tp = ax1.bar(xaxis, df["Total params"], label = "Total parameters", color = "red")
    ax1.set_xlabel("Models")
    ax1.set_ylabel("Total parameters")
    ax1.tick_params(axis = "x", labelrotation = 45)

    # Accuracy
    ax2 = ax1.twinx()
    acc = ax2.plot(xaxis, df["Kaggle Public score [%]"], label = "Kaggle Public score", color = "blue")
    ax2.set_ylabel("Accuracy in %")
    ax2.set_ylim(0.0, 100.0)

    lns = [tp] + acc  # Labels
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    plt.title("Total parameters and Accuracy")
    #fig.savefig("Total_params_and_Accuracy.png")
    plt.show()


def plot_ratio_accuracy(df):
    """This function plots to the trained models and the ratio of trainable parameters and the achieved accuracy on kaggle.

    Args:
        df (DataFrame): Models, total parameters, accuracy and ratio
    """

    fig, ax1 = plt.subplots()
    plt.subplots_adjust(bottom = 0.25)
    xaxis = df["Model"]
    df = df.sort_values(by = "Tp / KPs", ascending = True)  # sort df by ratio

    # Ratio
    r = ax1.plot(xaxis, df["Tp / KPs"], label = "Ratio", color = "red")
    ax1.set_xlabel("Models")
    ax1.set_ylabel("Ratio")
    ax1.tick_params(axis = "x", labelrotation = 45)

    # Accuracy
    ax2 = ax1.twinx()
    acc = ax2.plot(xaxis, df["Kaggle Public score [%]"], label = "Kaggle Public score", color = "blue")
    ax2.set_ylabel("Accuracy in %")
    ax2.set_ylim(0.0, 100.0)

    lns = r + acc  # Labels
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    plt.title("Ratio and Accuracy")
    #fig.savefig("Ratio_and_Accuracy.png")
    plt.show()


if (__name__ == "__main__"):
    # calculating the programme runtime
    Pr = ProgramRuntime()
    # Code here
    Pr.finish(print=True)

    # configures the hardware
    strategy = hardware_config("GPU")

    with strategy.scope():
        pass
        # Code here

    # read Acc.xlsx
    #df = pd.read_excel("./Portfolio3/Results/Acc.xlsx", sheet_name = "Tabelle3")
    #print(df)

    #plot_total_params_accuracy(df)
    #plot_ratio_accuracy(df)
    