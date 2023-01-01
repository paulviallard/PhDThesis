import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../chap-7-dis-mu/sourcecode")
from core.nd_data import NDData

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/slides_header_standalone.tex", "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)

plt.rcParams.update({
    "font.size": 8,
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

data = NDData("../chap-7-dis-mu/sourcecode/result.csv")
measure_list = data.keys("measure")
measure_dict = {
    "dist_fro": r"\distfro", "dist_l2": r"\distltwo",
    "param_norm": r"\paramnorm", "path_norm": r"\pathnorm",
    "sum_fro": r"\sumfro", "zero": r"\zero"}
dataset = "mnist"
hatch_list = [r"*", r".", r"\\", r"o", r"o-", r"O."]
color_list = [BLUE, CYAN, GREY, ORANGE, RED, MAGENTA]


# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 1, figsize=((0.4*14.0, 0.4*7.0)))

for i in range(len(measure_dict.keys())):
    measure = list(measure_dict.keys())[i]

    d = data.get("seeger_bound", "risk_test", measure=measure, data=dataset)

    seeger_bound = np.array(d["seeger_bound"])
    risk_test = np.array(d["risk_test"])

    j = np.argmin(seeger_bound, axis=0)
    seeger_bound_min = seeger_bound[i]
    risk_test_min = risk_test[i]

    ax.bar(i, risk_test_min, color=RED)
    ax.bar(i, seeger_bound_min, hatch=r"\\", fill=False, color=BLACK)

ax.set_xticks(np.arange(0, len(measure_dict)))
ax.set_xticklabels(measure_dict.values(), fontsize=7)

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/gap_{}.pdf".format(dataset), bbox_inches="tight")
plt.close(fig)
