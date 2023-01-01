import os
import sys
import itertools
import matplotlib.pyplot as plt

sys.path.append("../chap-6-dis-pra/sourcecode")
from core.nd_data import NDData

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/slides_header_standalone.tex", "r")
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

BLACK = "#000000"
BLUE = "#0077BB"
CYAN = "#009988"
GREEN = "#009988"
ORANGE = "#EE7733"
RED = "#CC3311"
MAGENTA = "#EE3377"
GREY = "#BBBBBB"

###############################################################################

if __name__ == "__main__":

    fig, ax = plt.subplots(1, 1, figsize=((1, 2.5)))

    ax.bar(0, 0.1, hatch=r"\\", color=RED)
    ax.bar(0, 0.2, hatch=r"\\", fill=False)
    plt.axis('off')

    os.makedirs("figures/", exist_ok=True)
    plt.savefig(f"figures/gap_legend.pdf", bbox_inches="tight")
