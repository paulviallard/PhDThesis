import sys
import os
import numpy as np
from os import makedirs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
            fig.subplots_adjust(wspace=0.31, hspace=0.2)
            gs = fig.add_gridspec(2, 5)

        if(k+1 == len(data_dict.keys()) and i % 2 == 0):
            ax = plt.subplot(gs[i//2, 1:3])
        else:
            ax = plt.subplot(gs[i//2, 2*(i % 2):2*((i % 2)+1)])

        eS_dict = {}
        dS_dict = {}
        for learner in learner_dict.keys():

            tmp_data = data.get(
                "eS", "dS",
                learner=learner, voter=voter, data=data_).to_numpy()
            tmp_mean = np.mean(tmp_data, axis=0)
            eS_dict[learner] = tmp_mean[0]
            dS_dict[learner] = tmp_mean[1]

        ax.set_title(data_dict[data_])

        # We create the C-Bound values
        e, d = np.meshgrid(
            np.linspace(0.0, 0.5-0.001, 300), np.linspace(0.0, 0.5-0.001, 300))
        e_ = np.linspace(0.0, 0.5-0.001, 400)
        cb = (1.0-((1.0-(2.0*e+d))**2.0)/(1.0-2.0*d))
        cond_1 = (2*e+d >= 1)
        cond_2 = (d >= 2*(np.sqrt(e)-e))
        cb = np.ma.array(cb, mask=cond_1+cond_2)

        gibbs = (2.0*e+d)
        gibbs = np.ma.array(gibbs, mask=(gibbs > cb)+cond_1+cond_2)

        cs = ax.contourf(e, d, cb, 20)
        ax.contourf(e, d, gibbs, 20, colors=BLACK, alpha=0.3)
        ax.plot([0.0, 0.25], [0.0, 0.5], c=BLACK, linestyle="dashed")
        ax.plot(e_, 2.0*(np.sqrt(np.minimum(0.25, e_))-e_),
                "black", linewidth=2)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 0.5)

        # We plot the results of the different algorithms
        ax.scatter(
            eS_dict["c-bound-joint"],
            dS_dict["c-bound-joint"],
            marker="d", color="black", s=100, alpha=1, zorder=100)
        ax.scatter(
            eS_dict["bound-joint"],
            dS_dict["bound-joint"],
            marker="^", color="black", s=100, alpha=1, zorder=100)
        ax.scatter(
            eS_dict["bound-risk"],
            dS_dict["bound-risk"],
            marker="*", color="black", s=100, alpha=1, zorder=100)
        if("cb-boost" in eS_dict):
            ax.scatter(
                eS_dict["cb-boost"],
                dS_dict["cb-boost"],
                marker="o", color="black", s=100, alpha=1, zorder=100)
        if("mincq" in eS_dict):
            ax.scatter(
                eS_dict["mincq"],
                dS_dict["mincq"],
                marker="x", color="black", s=100, alpha=1, zorder=100)

        ax.set_xlabel(" ")
        ax.set_ylabel(" ")

        if(i == 0):
            ax = plt.subplot(gs[:, 4])
            ax.set_axis_off()
            fig.colorbar(cs, ax=ax, shrink=2.0, orientation='vertical', label=r"C-Bound $\CBound_{\dS}(\Q)$")

        fig.savefig("figures/cbound_{}_{}_{}.pdf".format(voter, mode, j),
                    bbox_inches="tight")

        i += 1
        if(i == 4):
            i = 0
            j += 1


###############################################################################
