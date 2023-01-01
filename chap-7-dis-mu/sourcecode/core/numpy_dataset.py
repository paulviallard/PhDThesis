from h5py import File
from re import sub
import re
from os.path import isfile
from torch.utils.data import Dataset
import torch
import numpy as np
import copy


class NumpyDataset(Dataset):

    def __init__(self, x_y_dict):

        self._dataset_key = list(x_y_dict.keys())

        # Removing the mode in self._dataset_key
        new_dataset_key = {}
        for key in self._dataset_key:
            new_key = sub("_[^_]+$", "", key)
            if(new_key != key):
                new_dataset_key[new_key] = None
        self._dataset_key = list(new_dataset_key.keys())

        self._dataset_dict = copy.deepcopy(x_y_dict)
        self._mode = "train"

        self._class_size = None
        self._set_class_size()

    def is_mode(self, mode):
        if(self._dataset_key[0]+"_"+mode in self._dataset_dict):
            return True
        return False

    def set_mode(self, mode):
        # Setting the mode of the dataset
        self._mode = mode
        self._class_size = None
        self._set_class_size()

    def get_mode(self):
        # Getting the mode of the dataset
        return self._mode

    def get_mode_dataset(self, key):
        mode_key = key+"_"+self._mode
        return self._dataset_dict[mode_key]

    def __len__(self):
        # Getting the size of a dataset (of a given "mode")
        return len(self.get_mode_dataset("x"))

    def _set_class_size(self):
        if("y_"+self._mode in self._dataset_dict and self._class_size is None):
            self._class_size = len(np.unique(self.get_mode_dataset("y")))
        else:
            self._class_size = None

    def input_size(self):
        return list(self.get_mode_dataset("x").shape[1:])

    def __getitem__(self, i):
        # Getting each example for a given mode
        item_dict = {
            "mode": self._mode,
            "size": self.__len__(),
        }
        if(self._class_size is not None):
            item_dict["class_size"] = self._class_size

        for key in self._dataset_key:
            # If we have an example, we transform the example before
            item = self.get_mode_dataset(key)[i]
            if(isinstance(item, torch.Tensor)):
                item_dict[key] = item.clone().detach()
            else:
                item_dict[key] = torch.tensor(item)

        return item_dict
