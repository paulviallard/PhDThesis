import sys
import numpy as np
import pandas as pd
from os import makedirs

sys.path.append("sourcecode/")
from core.nd_data import NDData

###############################################################################

data = NDData("sourcecode/chap_4.csv")

binary = {"credit": "Credit", "heart": "Heart", "usvotes": "USVotes",
          "wdbc": "WDBC", "tictactoe": "TicTacToe", "svmguide": "SVMGuide",
          "haberman": "Haberman"}
multi = {"mnist": "MNIST", "fashion": "FashionMNIST", "pendigits": "Pendigits",
         "protein": "Protein", "shuttle": "Shuttle",
         "sensorless": "Sensorless", "glass": "Glass"}

learner_binary = {
    "mincq": r"\mincq",
    "cb-boost": r"\cbboost",
    "bound-risk": r"\algogermain",
    "bound-joint": r"\algomasegosa",
    "c-bound-joint": "Algorithm 4.3",
    "c-bound-seeger": "Algorithm 4.2",
    "c-bound-mcallester": "Algorithm 4.1",
}
learner_multi = dict(learner_binary)
del learner_multi["cb-boost"]
del learner_multi["mincq"]

voter_data_learner_list = [
    ["stump", binary, learner_binary, "binary",
     "l|cc||cc||cc||cc||cc||cc||cc"],
    ["tree", binary, learner_binary, "binary",
     "l|cc||cc||cc||cc||cc||cc||cc"],
    ["tree", multi, learner_multi, "multi", "l|cc||cc||cc||cc||cc"]
]


def latex_fun(x, r, c, data):

    if(c == ("", "")):
        return x

    c_risk = "$\\Risk_{\\dT}(\\MVQ)$"
    c_bound = r"Bound"

    # We get the bound and the risk
    risk = float(data.iloc[r][(c[0], c_risk)].split(",")[0])
    bound = float(data.iloc[r][(c[0], c_bound)].split(",")[0])

    data_list = data.iloc[r].to_list()
    data_list = data_list[1:]
    bound_list = []
    risk_list = []
    for i in range(len(data_list)):
        value = float(data_list[i].split(",")[0])
        if(i % 2 == 0):
            risk_list.append(value)
        else:
            bound_list.append(value)

    risk_list = np.sort(np.unique(risk_list))
    bound_list = np.sort(np.unique(bound_list))
    risk_min = risk_list[0]
    bound_min = bound_list[0]
    bound_2_min = bound_list[1]

    begin = ""
    end = ""
    if(bound_2_min == bound):
        begin += r"\underline{"
        end += r"}"
    if(risk_min == risk):
        begin += r"\textbf{"
        end += r"}"
    if(bound_min == bound):
        begin += r"\textit{"
        end += r"}"

    mean = float("{:.3f}".format(float(x.split(",")[0])))
    mean = str(mean).replace("0.", ".")
    if(len(mean.split(".")[1]) < 3):
        mean += "0"*(3-len(mean.split(".")[1]))
    std = float("{:.3f}".format(float(x.split(",")[1])))
    std = str(std).replace("0.", ".")
    if(len(std.split(".")[1]) < 3):
        std += "0"*(3-len(std.split(".")[1]))

    x = r"{}{} $\pm$ {}{}".format(begin, mean, std, end)
    return x


###############################################################################

for (voter, data_dict, learner_dict, mode, format) in voter_data_learner_list:

    tmp_all = None
    for learner in learner_dict.keys():

        tmp = None
        for data_ in data_dict.keys():

            tmp_data = data.get(
                "tT", "bound",
                learner=learner, voter=voter, data=data_).to_numpy()
            tmp_mean = np.mean(tmp_data, axis=0)
            tmp_std = np.std(tmp_data, axis=0)

            tmp_data = {
                "data": [data_dict[data_]],
                "tT": ["{},{}".format(tmp_mean[0], tmp_std[0])],
                "bound": ["{},{}".format(tmp_mean[1], tmp_std[1])]}
            tmp_data = pd.DataFrame(data=tmp_data)

            if(tmp is None):
                tmp = tmp_data
            else:
                tmp = pd.concat([tmp, tmp_data])

        arrays = [
            np.array(["",
                      learner_dict[learner], learner_dict[learner]]),
            np.array(["", r"$\Risk_{\dT}(\MVQ)$", r"Bound"]),
        ]
        tmp.columns = arrays

        if(tmp_all is None):
            tmp_all = tmp
        else:
            tmp_all = pd.merge(tmp, tmp_all, how="right", on=[("", "")])

    makedirs("tables/", exist_ok=True)
    with open("tables/"+mode+"-"+voter+".tex", "w") as f:
        f.write(NDData.to_latex(tmp_all, latex_fun, col_format=format))
