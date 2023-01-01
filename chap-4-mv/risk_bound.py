import sys
import os
import numpy as np
from os import makedirs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.append("sourcecode/")
from core.nd_data import NDData

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/header_standalone.tex", "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)

plt.rcParams.update({
    "font.size": 14,
    "text.usetex": True,
    "pgf.rcfonts": False,
    "text.latex.preamble": preamble,
    "pgf.preamble": preamble,
})

###############################################################################

WHITE = "#FFFFFF"
BLACK = "#000000"
BLUE = "#0077BB"
CYAN = "#009988"
GREEN = "#009988"
ORANGE = "#EE7733"
RED = "#CC3311"
MAGENTA = "#EE3377"
GREY = "#BBBBBB"

blue_cmap = LinearSegmentedColormap.from_list(
    "blue_cmap", [BLUE, ORANGE])

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
    ["stump", binary, learner_binary, "binary"],
    ["tree", binary, learner_binary, "binary"],
    ["tree", multi, learner_multi, "multi"]
]

###############################################################################
makedirs("figures/", exist_ok=True)

for (voter, data_dict, learner_dict, mode) in voter_data_learner_list:

    learner_list = []
    for learner in learner_dict.keys():
        learner_list.append(learner_dict[learner])

    i = 0
    j = 1
    for k in range(len(data_dict.keys())):
        data_ = list(data_dict.keys())[k]

        if(i == 0):
            fig = plt.figure(figsize=(10, 10))
            fig.subplots_adjust(wspace=0.31, hspace=0.5)
            gs = fig.add_gridspec(2, 4)

        if(k+1 == len(data_dict.keys()) and i % 2 == 0):
            ax = plt.subplot(gs[i//2, 1:3])
        else:
            ax = plt.subplot(gs[i//2, 2*(i % 2):2*((i % 2)+1)])

        risk_mean = []
        bound_mean = []
        risk_std = []
        bound_std = []
        for learner in learner_dict.keys():

            tmp_data = data.get(
                "tT", "bound",
                learner=learner, voter=voter, data=data_).to_numpy()
            tmp_mean = np.mean(tmp_data, axis=0)
            tmp_std = np.std(tmp_data, axis=0)
            risk_mean.append(tmp_mean[0])
            bound_mean.append(tmp_mean[1])
            risk_std.append(tmp_std[0])
            bound_std.append(tmp_std[1])

        ax.set_title(data_dict[data_])
        ax.bar(learner_list, bound_mean, yerr=bound_std,
               alpha=0.8, hatch="//", color=BLUE,
               error_kw=dict(lw=3, ecolor=BLACK, capsize=10, capthick=1))
        ax.bar(learner_list, risk_mean, yerr=risk_std,
               alpha=1, hatch="..", color=ORANGE,
               error_kw=dict(lw=3, ecolor=BLACK, capsize=10, capthick=1))

        ax.set_xticks(learner_list)
        ax.set_xticklabels(learner_list, fontsize=12, rotation=70)
        ax.set_ylim(bottom=0.0)

        fig.savefig("figures/{}_{}_{}.pdf".format(voter, mode, j),
                    bbox_inches="tight")

        i += 1
        if(i == 4):
            i = 0
            j += 1
