import sys
import os
import matplotlib.pyplot as plt
import numpy as np

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

###############################################################################

data = NDData("sourcecode/result.csv")
measure_dict = data.keys("measure")

measure_list = [
    {"dist_fro": r"\distfro", "dist_l2": r"\distltwo",
     "param_norm": r"\paramnorm", "path_norm": r"\pathnorm",
     "sum_fro": r"\sumfro", "zero": r"\zero"},
    {"dist_fro-aug": r"\distfroaug", "dist_l2-aug": r"\distltwoaug",
     "param_norm-aug": r"\paramnormaug", "path_norm-aug": r"\pathnormaug",
     "sum_fro-aug": r"\sumfroaug", "zero-aug": r"\zeroaug"}
]
dataset_dict = {"mnist": "MNIST", "fashion": "FashionMNIST"}

name_dict = {0: "", 1: "_aug"}

# --------------------------------------------------------------------------- #

for j in range(len(measure_list)):
    for dataset in dataset_dict.keys():
        fig, ax_list = plt.subplots(3, 2, figsize=((9.0, 14.0)))
        measure_dict = measure_list[j]

        for i in range(len(measure_dict.keys())):
            measure = list(measure_dict.keys())[i]

            ax = ax_list[i//2, i % 2]

            xtick_list = []
            min_bound_list = []
            max_bound_list = []
            mean_bound_list = []
            std_bound_list = []

            for depth in data.keys("depth"):
                for width in data.keys("width"):

                    d = data.get(
                        "seeger_bound", "risk_test", "alpha", measure=measure,
                        data=dataset, width=width, depth=depth)
                    d = d.sort_values(by=["alpha"])
                    xtick_list.append("{}/{}".format(3*(int(depth)+1), width))

                    bound = np.array(d["seeger_bound"])
                    min_bound_list.append(np.min(bound))
                    mean_bound_list.append(np.mean(bound))
                    std_bound_list.append(np.std(bound))
                    max_bound_list.append(np.max(bound))

            min_bound_list = np.array(min_bound_list)
            max_bound_list = np.array(max_bound_list)
            mean_bound_list = np.array(mean_bound_list)
            std_bound_list = np.array(std_bound_list)

            ax.set_title(
                r"{},\ \ $\comp(\h,\S)={}$".format(
                    dataset_dict[dataset], measure_dict[measure]))

            ax.plot(xtick_list, min_bound_list, "--", c=BLACK)
            ax.plot(xtick_list, max_bound_list, "--", c=BLACK)
            ax.plot(xtick_list, mean_bound_list, "-", c=BLUE)
            ax.fill_between(xtick_list, mean_bound_list-std_bound_list,
                            mean_bound_list+std_bound_list, alpha=0.2)

        os.makedirs("figures/", exist_ok=True)
        os.makedirs("figures/influence_depth/", exist_ok=True)
        fig.savefig("figures/influence_depth/influence_depth_{}{}.pdf".format(
            dataset, name_dict[j]), bbox_inches="tight")
        plt.close(fig)
