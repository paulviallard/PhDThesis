from argparse import ArgumentParser
import h5py
import numpy as np
import os
import pandas as pd
import random
import requests
import warnings

###############################################################################


def get_label(data):
    """
    Get the labels of the loaded dataset


    Parameters
    ----------
    data : DataFrame
        Pandas dataframe containing the loaded dataset

    Returns
    -------
    ndarray
        Vector of size N (N is the number of inputs) with the labels
    """
    label = data[8]
    label = label.astype(int)
    label = label.to_numpy()
    return label


def get_input(data):
    """
    Get the inputs of the loaded dataset

    Parameters
    ----------
    data : DataFrame
        Pandas dataframe containing the loaded dataset

    Returns
    -------
    ndarray
        Matrix of size NxM with the inputs
        (N is the number of examples & M is the number of features)
    """
    data = data.drop([8], axis=1)

    data[0].loc[data[0] == "M"] = 0.0
    data[0].loc[data[0] == "F"] = 1.0
    data[0].loc[data[0] == "I"] = 2.0

    input = data.to_numpy().astype(np.float32)

    # We rescale the data with a min-max scaling
    input_min_max = (input - input.min(axis=0))
    input_min_max /= (input.max(axis=0) - input.min(axis=0))
    return input_min_max


def get_train_test(input, label, ratio_test):
    """
    Get the train/test dataset

    Parameters
    ----------
    input: ndarray
        Matrix of size NxM with the inputs
        (N is the number of examples & M is the number of features)
    label: ndarray
        Vector of size N (N is the number of inputs) with the labels
    ratio_test: float
        Percentage of the original set as the test set

    Returns
    -------
    ndarray, ndarray, ndarray, ndarray
        Input matrices and label vectors for the train set and the test set
    """
    size_test = int(ratio_test*len(input))

    input_test = input[:size_test, :]
    input_train = input[size_test:, :]
    label_test = label[:size_test]
    label_train = label[size_test:]

    return input_train, input_test, label_train, label_test


def shuffle(input, label):
    """
    Shuffle the dataset

    Parameters
    ----------
    input: ndarray
        Matrix of size NxM with the inputs
        (N is the number of examples & M is the number of features)
    label: ndarray
        Vector of size N (N is the number of inputs) with the labels

    Returns
    -------
    ndarray, ndarray
        The shuffled matrix and vector
    """
    permutation = np.arange(input.shape[0])
    np.random.shuffle(permutation)
    input = input[permutation, :]
    label = label[permutation]
    return input, label


def save(path, input_train, input_test, label_train, label_test):
    """
    Save the train/test set in the h5 file

    Parameters
    ----------
    path : str
        Path of the h5 file to save
    input_train: ndarray
        Inputs of the train set
    input_test: ndarray
        Inputs of the test set
    label_train: ndarray
        Labels of the train set
    label_test: ndarray
        Labels of the test set
    """
    dataset_file = h5py.File(path, "w")

    dataset_file["x_train"] = input_train
    dataset_file["y_train"] = label_train
    dataset_file["x_test"] = input_test
    dataset_file["y_test"] = label_test


###############################################################################

def main():
    # We initialize the seeds
    np.random.seed(42)
    random.seed(42)
    warnings.filterwarnings("ignore")

    # We parse the arguments
    arg_parser = ArgumentParser(
        description="Generate letter dataset")
    arg_parser.add_argument(
        "--test", metavar="test", default=0.15, type=float,
        help="Proportion of the test set")

    arg_list = arg_parser.parse_args()
    ratio_test = arg_list.test

    # We download the dataset
    if(not(os.path.exists("data-abalone/"))
       or not(os.path.exists("data-abalone/abalone.data"))):

        if(not(os.path.exists("data-abalone"))):
            os.mkdir("data-abalone")

        r = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            + "abalone/abalone.data",
            allow_redirects=True)
        f = open("data-abalone/abalone.data", "wb")
        f.write(r.content)
        f.close()

    # We open, process and save the dataset

    data = pd.read_csv(
        "data-abalone/abalone.data", sep=",", header=None)

    label = get_label(data)
    input = get_input(data)

    input, label = shuffle(input, label)
    (input_train, input_test, label_train, label_test) = get_train_test(
        input, label, ratio_test)
    save("abalone.h5", input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
