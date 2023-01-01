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

data = NDData("sourcecode/chap_4_moons.csv")

learner_dict = {
    "bound-risk": r"\algogermain",
    "bound-joint": r"\algomasegosa",
    "c-bound-mcallester": "Algorithm 4.1",
    "c-bound-seeger": "Algorithm 4.2",
    "c-bound-joint": "Algorithm 4.3",
}

learner_plot_dict = {
    "bound-risk": {"c": BLUE, "linestyle": "dashed"},
    "bound-joint": {"c": BLACK, "linestyle": "dashed"},
    "c-bound-joint": {"c": GREEN},
    "c-bound-seeger": {"c": RED},
    "c-bound-mcallester": {"c": ORANGE}
}

learner_fill_dict = {
    "bound-risk": {"color": BLUE},
    "bound-joint": {"color": BLACK},
    "c-bound-joint": {"color": GREEN},
    "c-bound-seeger": {"color": RED},
    "c-bound-mcallester": {"color": ORANGE}
}

###############################################################################
makedirs("figures/", exist_ok=True)

fig = plt.figure(figsize=(5, 5))
fig.subplots_adjust(wspace=0.31, hspace=0.4)
gs = fig.add_gridspec(2, 1)

ax_size = plt.subplot(gs[0, 0])
ax_voter = plt.subplot(gs[1, 0])

seed_list = [i for i in range(10)]
size_train_list = [100, 500, 1000, 5000]
size_stump_list = [32, 64, 128]

for learner in learner_dict:

    time_list_mean = []
    time_list_std = []

    for size_train in size_train_list:

        data_list = ["moons-"+str(0.02).replace(".", "_")+"-"
                     + str(seed)+"-"+str(size_train)
                     for seed in seed_list]
        tmp_data = data.get(
            "time",
            learner=learner, data=data_list, nb_per_attribute=128).to_numpy()

        time_list_mean.append(np.mean(tmp_data, axis=0)[0])
        time_list_std.append(np.std(tmp_data, axis=0)[0])

    time_list_mean = np.array(time_list_mean)
    time_list_std = np.array(time_list_std)

    ax_size.plot(
        size_train_list, time_list_mean,
        label=learner_dict[learner], **learner_plot_dict[learner])
    ax_size.fill_between(
        size_train_list, time_list_mean+time_list_std,
        time_list_mean-time_list_std, alpha=0.1,
        **learner_fill_dict[learner])
    ax_size.set_ylabel(r"Time (in s)")
    ax_size.set_xlabel(r"Number of examples $\m$")

    ax_size.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.4),
        frameon=False, ncol=3, fontsize=12)

    time_list_mean = []
    time_list_std = []
    for size_stump in size_stump_list:

        data_list = ["moons-"+str(0.02).replace(".", "_")+"-"
                     + str(seed)+"-"+str(5000)
                     for seed in seed_list]
        tmp_data = data.get(
            "time",
            learner=learner, data=data_list,
            nb_per_attribute=size_stump).to_numpy()

        time_list_mean.append(np.mean(tmp_data, axis=0)[0])
        time_list_std.append(np.std(tmp_data, axis=0)[0])

    time_list_mean = np.array(time_list_mean)
    time_list_std = np.array(time_list_std)

    ax_voter.plot(
        size_stump_list, time_list_mean,
        **learner_plot_dict[learner])
    ax_voter.fill_between(
        size_stump_list, time_list_mean+time_list_std,
        time_list_mean-time_list_std, alpha=0.1,
        **learner_fill_dict[learner])
    ax_voter.set_ylabel(r"Time (in s)")
    ax_voter.set_xlabel(r"Number of Decision Stumps per Feature")

fig.savefig("figures/moons_time.pdf",
            bbox_inches="tight")
