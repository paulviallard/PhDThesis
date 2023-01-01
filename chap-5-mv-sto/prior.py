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
    "font.size": 11,
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

data = NDData("sourcecode/chap_5_prior.csv")

binary = {"credit": "Credit", "heart": "Heart", "usvotes": "USVotes",
          "wdbc": "WDBC", "tictactoe": "TicTacToe", "svmguide": "SVMGuide",
          "haberman": "Haberman"}
multi = {
    "mnist": "MNIST", "fashion": "FashionMNIST",
    "pendigits": "Pendigits",
    "protein": "Protein", "shuttle": "Shuttle",
    "sensorless": "Sensorless", "glass": "Glass"}

data_dict = dict(binary)
data_dict.update(multi)

learner_dict = {
    "bound-risk": r"\algogermain",
    "bound-joint": r"\algomasegosa",
    "bound-rand": r"\algolacasse",
    "c-bound-seeger": "Algorithm 4.2",
    "bound-sto": r"Algorithm 5.4",
}

learner_color_dict = {
    "bound-sto": BLUE,
    "bound-rand": ORANGE,
    "bound-joint": RED,
    "c-bound-seeger": GREEN,
    "bound-risk": BLACK
}

voter = "tree"
mode = "multi"
data_size = len(data_dict)

###############################################################################
makedirs("figures/", exist_ok=True)

learner_list = []
for learner in learner_dict.keys():
    learner_list.append(learner_dict[learner])

i = 0
j = 1
for k in range(len(data_dict.keys())):
    data_ = list(data_dict.keys())[k]

    if(i == 0 and data_size-k >= 4):
        fig = plt.figure(figsize=(10, 10*1.2))
        fig.subplots_adjust(wspace=0.4, hspace=0.5)
        gs = fig.add_gridspec(4, 3)
    elif(i == 0 and data_size-k < 4):
        fig = plt.figure(figsize=(10, 10*((data_size-k)/4)*1.2))
        fig.subplots_adjust(wspace=0.31, hspace=0.5)
        gs = fig.add_gridspec(data_size-k, 3)

    ax_tT = plt.subplot(gs[i, 0])
    ax_tS = plt.subplot(gs[i, 1])
    ax_bound = plt.subplot(gs[i, 2])

    for learner in learner_dict.keys():

        tmp_data = data.get(
            "sto_prior",
            learner=learner, voter=voter, data=data_)
        sto_prior_list = np.unique(tmp_data["sto_prior"].to_numpy())

        tT_mean = []
        tS_mean = []
        bound_mean = []
        tT_std = []
        tS_std = []
        bound_std = []

        for sto_prior in sto_prior_list:

            tmp_data = data.get(
                "tT", "tS", "bound",
                learner=learner, voter=voter, data=data_,
                sto_prior=sto_prior).to_numpy()

            tmp_mean = np.mean(tmp_data, axis=0)
            tmp_std = np.std(tmp_data, axis=0)
            tT_mean.append(tmp_mean[0])
            tS_mean.append(tmp_mean[1])
            bound_mean.append(tmp_mean[2])
            tT_std.append(tmp_std[0])
            tS_std.append(tmp_std[1])
            bound_std.append(tmp_std[2])

        bound_mean = np.array(bound_mean)
        bound_std = np.array(bound_std)
        tT_mean = np.array(tT_mean)
        tT_std = np.array(tT_std)
        tS_mean = np.array(tS_mean)
        tS_std = np.array(tS_std)

        sto_prior_list = np.array([float(s) for s in sto_prior_list])

        ax_tS.set_title(data_dict[data_])

        ax_bound.plot(
            sto_prior_list, bound_mean,
            c=learner_color_dict[learner], label=learner_dict[learner])
        ax_bound.fill_between(
            sto_prior_list, bound_mean+bound_std,
            bound_mean-bound_std, alpha=0.1,
            color=learner_color_dict[learner])
        ax_bound.set_ylabel(r"Bound Value")
        ax_bound.set_xlabel(r"$\sparamDirP$")
        ax_bound.set_xscale('log')

        ax_tT.plot(
            sto_prior_list, tT_mean,
            c=learner_color_dict[learner], label=learner_dict[learner])
        ax_tT.fill_between(
            sto_prior_list, tT_mean+tT_std,
            tT_mean-tT_std, alpha=0.1,
            color=learner_color_dict[learner])
        ax_tT.set_ylabel(r"Test Risk")
        ax_tT.set_xlabel(r"$\sparamDirP$")
        ax_tT.set_xscale('log')

        ax_tS.plot(
            sto_prior_list, tS_mean,
            c=learner_color_dict[learner], label=learner_dict[learner])
        ax_tS.fill_between(
            sto_prior_list, tS_mean+tS_std,
            tS_mean-tS_std, alpha=0.1,
            color=learner_color_dict[learner])
        ax_tS.set_ylabel(r"Empirical Risk")
        ax_tS.set_xlabel(r"$\sparamDirP$")
        ax_tS.set_xscale('log')

    if(i == 0):
        ax_tS.legend(
            loc='upper center', bbox_to_anchor=(0.4, 1.4),
            frameon=False, ncol=5, fontsize=14)

    fig.savefig("tmp.png")
    os.remove("tmp.png")
    if(ax_bound.get_ylim()[1] > 1.0):
        ax_bound.set_ylim(top=1.0)

    fig.savefig("figures/prior_{}.pdf".format(j),
                bbox_inches="tight")

    i += 1
    if(i == 4):
        i = 0
        j += 1
