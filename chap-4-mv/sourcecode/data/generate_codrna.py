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
        description="Generate codrna dataset")
    arg_parser.parse_args()

    # We download the dataset
    if(not(os.path.exists("data-codrna/"))
       or not(os.path.exists("data-codrna/cod-rna"))
       or not(os.path.exists("data-codrna/cod-rna.t"))
       or not(os.path.exists("data-codrna/cod-rna.r"))):

        if(not(os.path.exists("data-codrna"))):
            os.mkdir("data-codrna")

        r = requests.get(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
            + "cod-rna",
            allow_redirects=True)
        f = open("data-codrna/cod-rna", "wb")
        f.write(r.content)
        f.close()

        r = requests.get(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
            + "cod-rna.t",
            allow_redirects=True)
        f = open("data-codrna/cod-rna.t", "wb")
        f.write(r.content)
        f.close()

        r = requests.get(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
            + "cod-rna.r",
            allow_redirects=True)
        f = open("data-codrna/cod-rna.r", "wb")
        f.write(r.content)
        f.close()

    # We open, process and save the dataset
    data_list = load_svmlight_files(
        ["data-codrna/cod-rna",
         "data-codrna/cod-rna.t",
         "data-codrna/cod-rna.r"])
    input_train = pd.DataFrame(data_list[0].toarray())
    label_train = pd.DataFrame(data_list[1])
    input_val = pd.DataFrame(data_list[2].toarray())
    label_val = pd.DataFrame(data_list[3])
    input_test = pd.DataFrame(data_list[4].toarray())
    label_test = pd.DataFrame(data_list[5])

    input_train = input_train.append(input_val)
    label_train = label_train.append(label_val)

    input_train_test = input_train.append(input_test)
    input_train_test = get_input(input_train_test)
    input_train = input_train_test[:len(input_train)]
    input_test = input_train_test[len(input_train):]

    label_train = get_label(label_train)
    label_test = get_label(label_test)

    input_train, label_train = shuffle(input_train, label_train)
    input_test, label_test = shuffle(input_test, label_test)

    save("codrna.h5", input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
