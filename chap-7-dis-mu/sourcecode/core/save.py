import pandas as pd
import numpy as np


def to_latex(data, float_format="{:0.2f}"):
    s = "\\begin{tabular}\n\\toprule\n"

    col_list = data.columns.to_list()
    index_list = data.index.to_list()

    s += " "
    for col in col_list:
        s += "& {} ".format(col)
    s += "\\\\\n"

    for index in index_list:
        s += "{} ".format(index)
        val_list = data.loc[index].to_list()
        for val in val_list:
            if(isinstance(val, float)):
                s += ("& "+float_format+" ").format(val)
            else:
                s += "& {} ".format(val)
        s += "\\\\\n"

    print(index_list)


    s += "\\bottomrule\n\\end{tabular}\n"

    return s
