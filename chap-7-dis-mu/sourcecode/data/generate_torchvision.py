import sys
import os
import h5py
import random
import torch
import warnings
import torchvision
import torchvision.datasets
import numpy as np
from torch.utils.data import DataLoader
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.tools import call_fun


def get_label(label):
    """
    Get the labels of the loaded labels


    Parameters
    ----------
    label : torch.Tensor
        Tensor containing the loaded labels

    Returns
    -------
    ndarray
        Vector of size N (N is the number of inputs) with the labels
    """
    label = label.numpy().astype(int)
    return label


def get_input(input):
    """
    Get the inputs of the loaded inputs

    Parameters
    ----------
    input : torch.Tensor
        Tensor containing the loaded inputs

    Returns
    -------
    ndarray
        Matrix of size NxM with the inputs
        (N is the number of examples & M is the number of features)
    """
    input = input.numpy().astype(np.float32)
    return input


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
    torch.manual_seed(42)
    warnings.filterwarnings("ignore")

    # We parse the arguments
    arg_parser = ArgumentParser(
        description="generate a torchvision dataset")
    arg_parser.add_argument(
        "dataset", metavar="dataset", type=str,
        help="name of the dataset"
    )
    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="path of the h5 dataset file"
    )
    arg_list = arg_parser.parse_args()

    # We get the arguments
    dataset = arg_list.dataset
    path = arg_list.path

    # Loading a folder as dataset
    if(os.path.exists(dataset)):
        input_label_train = torchvision.datasets.ImageFolder(
            root="./"+dataset+"/train",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        )
        input_label_test = torchvision.datasets.ImageFolder(
            root="./"+dataset+"/test",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        )
        test_size = len(input_label_test)
        train_size = len(input_label_train)

    # Loading a torchvision dataset
    else:
        dataset_fun = None
        _locals = locals()
        exec("dataset_fun = torchvision.datasets."+str(dataset),
             globals(), _locals)
        dataset_fun = _locals["dataset_fun"]
        kwargs = {
            "root": "./data-"+dataset,
            "train": True,
            "download": True,
            "split": "train",
            "transform": torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        }

        input_label_train = call_fun(dataset_fun, kwargs)
        if("train" in kwargs):
            kwargs["train"] = False
        if("split" in kwargs):
            kwargs["split"] = "test"
        input_label_test = call_fun(dataset_fun, kwargs)

        test_size = input_label_test.data.shape[0]
        train_size = input_label_train.data.shape[0]

    # We get the train and test data
    train_loader = DataLoader(
        input_label_train,
        batch_size=train_size)
    test_loader = DataLoader(
        input_label_test, batch_size=test_size)
    input_label_train = list(train_loader)
    input_label_test = list(test_loader)
    input_train = input_label_train[0][0]
    label_train = input_label_train[0][1]
    input_test = input_label_test[0][0]
    label_test = input_label_test[0][1]

    input_train = get_input(input_train)
    input_test = get_input(input_test)
    label_train = get_label(label_train)
    label_test = get_label(label_test)

    save(path, input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
