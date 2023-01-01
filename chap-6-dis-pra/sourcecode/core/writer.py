import os
import torch
import glob


class Writer():

    def __init__(self, path, load=True, erase=True):
        self._folder_path = path
        self.erase = True
        os.makedirs("{}".format(self._folder_path), exist_ok=True)
        self.__get_folder_list()

    def __get_folder_list(self):

        self.folder_dict = {}
        path_list = glob.glob(
            "{}/**/*.pkl".format(self._folder_path), recursive=True)

        if(len(path_list) > 0):
            path = path_list[0]
            path = path.replace("{}/".format(self._folder_path), "")
            path = path.replace("/writer.pkl", "")
            folder_list = path.split("/")
            for folder in folder_list:
                folder = folder.split("=")[0]
                self.folder_dict[folder] = None

    def __update_folder_list(self, folder_list):
        updated = False
        flag_updated = False
        for folder in folder_list:
            if(folder not in self.folder_dict):
                if(not(flag_updated) and len(self.folder_dict) > 0):
                    updated = True
                flag_updated = True

                self.folder_dict[folder] = None
        return updated

    def __get_folder_path(self, folder_value_dict):

        folder_list = list(self.folder_dict.keys())
        folder_list.sort()

        folder_path = self._folder_path+"/"
        for folder in folder_list:
            if(folder in folder_value_dict
               and folder_value_dict[folder] is not None):
                folder_path += "{}={}/".format(
                    folder, folder_value_dict[folder])
            else:
                folder_path += "{}=/".format(folder)

        return folder_path

    def __get_folder_value_dict(self, folder_path):
        folder_value_dict = dict(self.folder_dict)
        for folder_value in folder_path.split("/"):
            folder = folder_value.split("=")[0]
            value = folder_value.split("=")[1]
            folder_value_dict[folder] = value
        return folder_value_dict

    def __update_all_folder(self):
        path_list = glob.glob(
            "{}/**/*.pkl".format(self._folder_path), recursive=True)
        for path in path_list:
            path = path.replace("/writer.pkl", "")
            old_path = path+"/"
            path = path.replace("{}/".format(self._folder_path), "")
            folder_value_dict = self.__get_folder_value_dict(path)

            new_path = self.__get_folder_path(folder_value_dict)

            os.makedirs(new_path, exist_ok=True)
            os.rename(old_path+"writer.pkl", new_path+"writer.pkl")
            rm_dir_ok = True
            parent_path = os.path.abspath(old_path)
            while rm_dir_ok:
                try:
                    os.rmdir(parent_path)
                    parent_path = os.path.abspath(
                        os.path.join(parent_path, ".."))
                except OSError:
                    rm_dir_ok = False

    def open(self, **kwargs):
        folder_value_dict = kwargs
        folder_list = list(kwargs.keys())
        for folder in folder_list:
            folder_value_dict[folder] = str(folder_value_dict[folder])

        updated = self.__update_folder_list(folder_list)
        if(updated):
            self.__update_all_folder()
        self.path = self.__get_folder_path(folder_value_dict)
        os.makedirs("{}".format(self.path), exist_ok=True)
        self.load()

    def write(self, key, value):
        if(self.erase):
            self.erase = False
            self.file_dict = {}
        if(key not in self.file_dict):
            self.file_dict[key] = []
        self.file_dict[key].append(value)

    def save(self):
        torch.save(self.file_dict, self.path+"writer.pkl")

    def update(self, update_dict):
        self.file_dict.update(update_dict)
        self.save()

    def load(self):
        assert os.path.exists(self.path)

        f = open(self.path+"writer.pkl", "a+")
        f.close()
        if(os.path.getsize(self.path+"writer.pkl") > 0):
            self.file_dict = torch.load(self.path+"writer.pkl")
        else:
            self.file_dict = {}

    def __contains__(self, key):
        return key in self.file_dict

    def __getitem__(self, key):
        if(isinstance(self.file_dict[key], list)
           and len(self.file_dict[key]) == 1):
            return self.file_dict[key][0]
        return self.file_dict[key]
