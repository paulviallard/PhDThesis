import re
import os
import glob
import time
import numpy as np
import pandas as pd


class NDData():

    def __init__(self, data=None):
        self.path_data = None
        self.got_lock = False
        self.pid = os.getpid()

        if(data is not None):
            if(isinstance(data, pd.DataFrame)):
                self.data = data.copy(deep=True)
            elif(isinstance(data, str)):

                self.path_data = data

                glob_list = glob.glob(data+".lock.*")
                if(not(os.path.exists(data))
                   and len(glob_list) == 0):
                    tmp_f = open(data, 'a+')
                    tmp_f.close()

                # Get the lock
                while not(self.got_lock):
                    try:
                        os.replace(data, data+".lock."+str(self.pid))
                        self.got_lock = True
                    except OSError:
                        time.sleep(0.2)

                    # Read the csv file
                    try:
                        self.data = pd.read_csv(
                            data+".lock."+str(self.pid), index_col=0)
                    except pd.errors.EmptyDataError:
                        self.data = pd.DataFrame()
                    except FileNotFoundError:
                        self.got_lock = False
                        time.sleep(0.2)
            else:
                raise ValueError("data must be a pd.DataFrame or str")
        else:
            self.data = pd.DataFrame()

        self._init_hidden_data()
        self._init_index()

    def _init_hidden_data(self):
        self._hidden_data = self.data.copy(deep=True)
        for i, index_ in enumerate(list(self.data.index)):
            index = self.__interpret_index(index_)
            for col, val in index:
                if(col not in self._hidden_data):
                    self._hidden_data[col] = np.nan
                self._hidden_data.loc[index_, col] = val

    def _init_index(self):
        self.index_dict = {}
        for i, index_ in enumerate(list(self._hidden_data.index)):
            index = self.__interpret_index(index_)
            for key, val in index:
                if(key not in self.index_dict):
                    self.index_dict[key] = {val: [i]}
                elif(val not in self.index_dict[key]):
                    self.index_dict[key][val] = [i]
                else:
                    self.index_dict[key][val].append(i)

    def __rename_dash(self, var):
        var = var.replace("-", "_")
        return var

    def __create_index(self, index_dict):
        index = ""
        for key in sorted(list(index_dict.keys())):
            index += "{}={}/".format(
                key, index_dict[key])
        index = index[:-1]
        return index

    def __interpret_index(self, index):
        key_val_dict = []
        for key_val in index.split("/"):
            key_val = re.split("[=,]", key_val)
            if(len(key_val) == 2):
                key = key_val[0]
                key = self.__rename_dash(key)
                val = key_val[1]
                key_val_dict.append((key, val))
            elif(len(key_val) > 1 and len(key_val) % 2 == 1):
                dict_name = key_val[0]
                dict_name = self.__rename_dash(dict_name)
                for i in range(len(key_val)//2):
                    key = key_val[2*i+1]
                    key = self.__rename_dash(key)
                    val = key_val[2*i+2]
                    key_val_dict.append((dict_name+"__"+key, val))

        return tuple(key_val_dict)

    def __str__(self):
        return str(self.data)

    def keys(self, key=None, sort=None):
        if(key is None):
            key_list = list(self.index_dict.keys())
        else:
            key_list = list(self.index_dict[key].keys())
        key_list.sort(key=sort)
        return key_list

    def to_numeric(self, col):
        self._hidden_data[col] = pd.to_numeric(self._hidden_data[col])

    def get(self, *args, **kwargs):

        # Show columns
        data = self._hidden_data[list(args)].copy(deep=True)
        row_set = set(range(len(data)))

        for key, val in kwargs.items():

            if(not(isinstance(val, list))):
                val = [val]

            new_row_set = set()
            if(isinstance(val, list)):
                val_list = val
                for val in val_list:
                    val = str(val)
                    try:
                        new_row_set = (
                            set(self.index_dict[key][val]) | new_row_set)
                    except KeyError:
                        pass
            row_set = row_set & new_row_set

        data = data.iloc[list(row_set)]
        data = data.sort_index()
        data = data.reset_index()
        data = data.drop(columns=["index"])
        return data

    def set(self, col_dict, index_dict, erase=False):
        # We modify the given index
        for key in self.index_dict.keys():
            if(key not in index_dict):
                index_dict[key] = None
        index_name = self.__create_index(index_dict)

        # We modify all the other index
        for old_index in self.data.index:
            old_index_dict = dict(self.__interpret_index(old_index))
            for index in index_dict:
                if(index not in old_index_dict):
                    old_index_dict[index] = None
            new_index = self.__create_index(old_index_dict)
            self.data = self.data.rename(index={old_index: new_index})

        # If the index exists
        if(index_name in self.data.index):

            # for all columns in the dictionary
            for column, value in col_dict.items():

                # We create the column in the csv file if it does not exist
                if(column not in self.data.columns):
                    self.data[column] = np.nan

                # We replace the value if erase is True
                if(np.isnan(self.data.loc[index_name][column])
                   or (not(np.isnan(self.data.loc[index_name][column]))
                       and erase)):
                    self.data.at[index_name, column] = value

        # Otherwise, we create the row and insert the values
        old_set = set(self.data.columns)
        new_set = set(col_dict.keys())
        new_column = sorted(list(new_set))
        new_set = sorted(list(old_set.difference(new_set)))

        # We create the new columns
        for column in new_set:
            col_dict[column] = np.nan
        for column in new_column:
            if column not in self.data:
                self.data[column] = np.nan

        # We add the values
        self.data.loc[index_name] = col_dict

        # Reinit the data
        self._init_hidden_data()
        self._init_index()

    def save(self, path=None):
        if(path is not None):
            self.data.to_csv(path)
        else:
            if(self.path_data is not None):
                self.data.to_csv(self.path_data+".lock."+str(self.pid))
            else:
                raise ValueError("self.data_path must be not None")

    def __del__(self):
        # Release lock
        if(self.got_lock):
            os.replace(self.path_data+".lock."+str(self.pid), self.path_data)
            self.got_lock = False

    @staticmethod
    def to_latex(data, fun, col_name_list=None, col_format="l"):
        data = data.copy(deep=True)
        copy_data = data.copy(deep=True)
        for i in list(data.index):
            for col in list(data.columns):
                data.loc[i, col] = fun(data.loc[i, col], i, col, copy_data)
        if(col_name_list is not None):
            data.columns = col_name_list

        s = data.style
        return s.to_latex(
            column_format=col_format, hrules=True)
