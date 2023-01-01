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
    "font.size": 10,
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
measure_list = data.keys("measure")

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

            mcallester_bound_list = []
            seeger_bound_list = []
            risk_train_list = []
            risk_test_list = []

            for depth in data.keys("depth"):
                for width in data.keys("width"):

                    d = data.get("mcallester_bound", "seeger_bound",
                                 "risk_train", "risk_test",
                                 measure=measure, data=dataset, width=width,
                                 depth=depth)

                    mcallester_bound = np.array(d["mcallester_bound"])
                    seeger_bound = np.array(d["seeger_bound"])
                    risk_train = np.array(d["risk_train"])
                    risk_test = np.array(d["risk_test"])

                    k = np.argmin(seeger_bound)
                    mcallester_bound_list.append(mcallester_bound[k])
                    seeger_bound_list.append(seeger_bound[k])
                    risk_train_list.append(risk_train[k])
                    risk_test_list.append(risk_test[k])

            mcallester_bound_list = np.array(mcallester_bound_list)
            seeger_bound_list = np.array(seeger_bound_list)
            risk_train_list = np.array(risk_train_list)
            risk_test_list = np.array(risk_test_list)

            ax = ax_list[i//2, i % 2]
            ax.set_title(
                (r"{}, $\comp(\h,\S)={}$"
                 ).format(dataset_dict[dataset], measure_dict[measure]))

            ax.scatter(seeger_bound_list, risk_test_list, zorder=10, s=15)
            ax.scatter(mcallester_bound_list, risk_test_list, zorder=10,
                       s=15, c=BLACK, marker="^")

            ax.scatter(risk_train_list, risk_test_list, zorder=10,
                       s=15, marker="s")

            for i in range(len(risk_train_list)):
                ax.plot(
                    [risk_train_list[i], mcallester_bound_list[i]],
                    [risk_test_list[i], risk_test_list[i]], c="gray")

            # To compute the lim
            fig.savefig("tmp.png")
            os.remove("tmp.png")
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            ax.plot([-1, 1], [-1, 1], "--", color="black")

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        os.makedirs("figures/", exist_ok=True)
        os.makedirs("figures/gap/", exist_ok=True)
        fig.savefig("figures/gap/gap_{}{}.pdf".format(dataset, name_dict[j]),
                    bbox_inches="tight")
        plt.close(fig)
