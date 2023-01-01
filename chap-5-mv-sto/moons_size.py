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

data = NDData("sourcecode/chap_5_moons.csv")

seed_list = [i for i in range(10)]
size_train_list = [100, 500, 1000, 5000]

learner_dict = {
    "exact": {"risk": "exact", },
    "mc-1": {"risk": "MC", "mc_draws": 1},
    "mc-10": {"risk": "MC", "mc_draws": 10},
    "mc-100": {"risk": "MC", "mc_draws": 100},
}

learner_plot_dict = {
    "exact": {"c": BLUE, "label": r"Exact"},
    "mc-1": {"c": ORANGE, "linestyle": "dashed", "label": r"MC -- 1"},
    "mc-10": {"c": GREEN, "linestyle": "dotted", "label": r"MC -- 10"},
    "mc-100": {"c": RED, "linestyle": "-.", "label": r"MC -- 100"},
}

learner_fill_dict = {
    "exact": {"color": BLUE},
    "mc-1": {"color": ORANGE},
    "mc-10": {"color": GREEN},
    "mc-100": {"color": RED},
}

###############################################################################
makedirs("figures/", exist_ok=True)

fig, ax_list = plt.subplots(2, 2, figsize=(11, 6))
fig.subplots_adjust(wspace=0.25, hspace=0.3)

for learner in learner_dict.keys():

    tS_mean = []
    tT_mean = []
    bound_mean = []
    time_mean = []
    tS_std = []
    tT_std = []
    bound_std = []
    time_std = []

    for size_train in size_train_list:

        data_list = ["moons-"+str(0.02).replace(".", "_")+"-"
                     + str(seed)+"-"+str(size_train)
                     for seed in seed_list]
        learner_dict[learner]["data"] = data_list

        tmp_data = data.get(
            "tS", "tT", "bound", "time", **learner_dict[learner]).to_numpy()
        tmp_mean = np.mean(tmp_data, axis=0)
        tmp_std = np.std(tmp_data, axis=0)

        tS_mean.append(tmp_mean[0])
        tT_mean.append(tmp_mean[1])
        bound_mean.append(tmp_mean[2])
        time_mean.append(tmp_mean[3])
        tS_std.append(tmp_std[0])
        tT_std.append(tmp_std[1])
        bound_std.append(tmp_std[2])
        time_std.append(tmp_mean[3])

    tS_mean = np.array(tS_mean)
    tT_mean = np.array(tT_mean)
    bound_mean = np.array(bound_mean)
    time_mean = np.array(time_mean)
    tS_std = np.array(tS_std)
    tT_std = np.array(tT_std)
    bound_std = np.array(bound_std)
    time_std = np.array(time_std)

    ax_list[0, 0].plot(
        size_train_list, tS_mean, **learner_plot_dict[learner])
    ax_list[0, 0].fill_between(
        size_train_list, tS_mean-tS_std,
        tS_mean+tS_std, alpha=0.1, **learner_fill_dict[learner])
    ax_list[0, 0].set_xlabel(r"Number of examples $\m$")
    ax_list[0, 0].set_ylabel(
        r"Empirical Risk $\EE_{\Q\sim\hyperQ}\Risk_{\dS}(\MVQ)$")
    ax_list[0, 0].set_xlim(100, 5000)

    ax_list[0, 1].plot(
        size_train_list, tT_mean, **learner_plot_dict[learner])
    ax_list[0, 1].fill_between(
        size_train_list, tT_mean-tT_std, tT_mean+tT_std, alpha=0.1,
        **learner_fill_dict[learner])
    ax_list[0, 1].set_xlabel(r"Number of examples $\m$")
    ax_list[0, 1].set_ylabel(
        r"Test Risk $\EE_{\Q\sim\hyperQ}\Risk_{\dT}(\MVQ)$")
    ax_list[0, 1].set_xlim(100, 5000)

    ax_list[1, 0].plot(
        size_train_list, bound_mean, **learner_plot_dict[learner])
    ax_list[1, 0].fill_between(
        size_train_list, bound_mean-bound_std, bound_mean+bound_std, alpha=0.1,
        **learner_fill_dict[learner])
    ax_list[1, 0].set_xlabel(r"Number of examples $\m$")
    ax_list[1, 0].set_ylabel(r"Bound Value")
    ax_list[1, 0].set_xlim(100, 5000)

    ax_list[1, 1].plot(
        size_train_list, time_mean, **learner_plot_dict[learner])
    ax_list[1, 1].fill_between(
        size_train_list, time_mean-time_std, time_mean+time_std, alpha=0.1,
        **learner_fill_dict[learner])
    ax_list[1, 1].set_xlabel(r"Number of examples $\m$")
    ax_list[1, 1].set_ylabel(r"Time (in s)")
    ax_list[1, 1].set_xlim(100, 5000)

    ax_list[0, 0].legend(
        loc='upper center',
        bbox_to_anchor=(1.0, 1.25),
        frameon=False, ncol=5, fontsize=14)

    fig.savefig("figures/moons_size.pdf", bbox_inches="tight")
