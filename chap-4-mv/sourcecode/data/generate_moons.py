from argparse import ArgumentParser
import h5py
import numpy as np
import random
import warnings
from sklearn import datasets

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
        Vector of size N (N is the number of examples) with the labels
    """
    label = 2*label-1
    label = label.astype(int)
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
    # We rescale the data with a min-max scaling
    input = input.astype(np.float32)
    input_min_max = (input - input.min(axis=0))
    input_min_max /= (input.max(axis=0) - input.min(axis=0))
    return input_min_max


def get_train_test(input, label, size_test):
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

    # We parse the arguments
    arg_parser = ArgumentParser(
        description="Generate moons dataset")

    arg_parser.add_argument(
        "--size_train", metavar="size", default=1000, type=int,
        help="Size of the training dataset")
    arg_parser.add_argument(
        "--noise", metavar="noise", default=0.1, type=float,
        help="Noise of the moons dataset")
    arg_parser.add_argument(
        "--size_test", metavar="size_test", default=1000, type=int,
        help="Size of the test set")
    arg_parser.add_argument(
        "--seed", metavar="seed", default=0, type=int,
        help="The seed")
    arg_parser.add_argument(
        "--path", metavar="path", default="moons.h5", type=str,
        help="The seed")

    arg_list = arg_parser.parse_args()
    size_train = arg_list.size_train
    noise = arg_list.noise
    size_test = arg_list.size_test
    seed = arg_list.seed
    path = arg_list.path

    # We initialize the seeds
    np.random.seed(seed)
    random.seed(seed)
    warnings.filterwarnings("ignore")

    # We generate the dataset
    input, label = datasets.make_moons(
        n_samples=size_train+size_test, noise=noise)

    label = get_label(label)
    input = get_input(input)

    input, label = shuffle(input, label)
    (input_train, input_test, label_train, label_test) = get_train_test(
        input, label, size_test)
    save(path, input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
