from argparse import ArgumentParser
import h5py
import numpy as np
import os
import pandas as pd
import random
import requests
import warnings
from sklearn.datasets import load_svmlight_files

###############################################################################


def get_label(label):
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
    label = label.astype(int)
    label = label.to_numpy()
    label = (2*label-1).astype(int)
    label = label[:, 0]
    return label


def get_input(input):
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
    input = input.to_numpy().astype(np.float32)

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
        description="Generate phishing dataset")
    arg_parser.add_argument(
        "--test", metavar="test", default=0.15, type=float,
        help="Proportion of the test set")
    arg_list = arg_parser.parse_args()
    ratio_test = arg_list.test

    # We download the dataset
    if(not(os.path.exists("data-phishing/"))
       or not(os.path.exists("data-phishing/phishing"))):

        if(not(os.path.exists("data-phishing"))):
            os.mkdir("data-phishing")

        r = requests.get(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
            + "phishing",
            allow_redirects=True)
        f = open("data-phishing/phishing", "wb")
        f.write(r.content)
        f.close()

    # We open, process and save the dataset
    data_list = load_svmlight_files(["data-phishing/phishing"])
    input = pd.DataFrame(data_list[0].toarray())
    label = pd.DataFrame(data_list[1])

    input = get_input(input)
    label = get_label(label)

    input, label = shuffle(input, label)
    (input_train, input_test, label_train, label_test) = get_train_test(
        input, label, ratio_test)
    save("phishing.h5", input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
