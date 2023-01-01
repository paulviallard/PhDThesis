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
        Vector of size N (N is the number of examples) with the labels
    """
    label = data[16]
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
    example = data.drop([16], axis=1)
    # We rescale the data with a min-max scaling
    example = example.to_numpy().astype(np.float32)
    example_min_max = (example - example.min(axis=0))
    example_min_max /= (example.max(axis=0) - example.min(axis=0))
    return example_min_max


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

    # We download the dataset
    if(not(os.path.exists("data-pendigits/"))
       or not(os.path.exists("data-pendigits/pendigits.tra"))
       or not(os.path.exists("data-pendigits/pendigits.tes"))):

        if(not(os.path.exists("data-pendigits"))):
            os.mkdir("data-pendigits")

        r_1 = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            + "pendigits/pendigits.tra", allow_redirects=True)
        r_2 = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            + "pendigits/pendigits.tes", allow_redirects=True)
        f_1 = open("data-pendigits/pendigits.tra", "wb")
        f_2 = open("data-pendigits/pendigits.tes", "wb")
        f_1.write(r_1.content)
        f_2.write(r_2.content)
        f_1.close()
        f_2.close()

    # We open, process and save the dataset
    data = pd.read_csv(
        "data-pendigits/pendigits.tra", sep=",", header=None)
    data_test = pd.read_csv(
        "data-pendigits/pendigits.tes", sep=",", header=None)

    label_train = get_label(data)
    label_test = get_label(data_test)

    data_train_test = data.append(data_test)
    input_train_test = get_input(data_train_test)
    input_train = input_train_test[:len(label_train), :]
    input_test = input_train_test[len(label_train):, :]

    save("pendigits.h5", input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
