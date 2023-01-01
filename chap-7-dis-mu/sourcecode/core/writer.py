import os
import torch


class Writer():

    def __init__(self, path, load=True):
        self.file_dict = {}
        self.path = path
        if(load):
            self.load()

    def write(self, key, value):
        if(key not in self.file_dict):
            self.file_dict[key] = []
        self.file_dict[key].append(value)
        self.save()

    def save(self):
        torch.save(self.file_dict, self.path)

    def update(self, update_dict):
        self.file_dict.update(update_dict)
        self.save()

    def load(self):
        if(os.path.exists(self.path)):
            self.file_dict = torch.load(self.path)

    def __contains__(self, key):
        return key in self.file_dict

    def __getitem__(self, key):
        if(isinstance(self.file_dict[key], list)
           and len(self.file_dict[key]) == 1):
            return self.file_dict[key][0]
        return self.file_dict[key]
