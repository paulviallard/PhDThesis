import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/header_standalone.tex", "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)

plt.rcParams.update({
    "font.size": 12,
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

scatter_cmap = LinearSegmentedColormap.from_list(
    "scatter_cmap", [BLUE, RED])
scatter_bg_cmap = LinearSegmentedColormap.from_list(
    "scatter_bg_cmap", [BLUE, WHITE])

###############################################################################

seed = 0
rng = default_rng(seed=seed)

y_1 = rng.normal(0.0, 1.0, size=(10,))
y_2 = rng.normal(0.0, 1.0, size=(10,))

# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 1, figsize=(10, 2))

x = np.arange(0, 10)
f_1 = interp1d(x, y_1, kind="quadratic")
f_2 = interp1d(x, y_2, kind="quadratic")

x = np.arange(0, 9, 0.1)
y_1, y_2 = f_1(x), f_2(x)

l1 = ax.plot(x, y_1, c=BLUE, label=r"$\loss(\h_{\S}, (\x, \y))$")
l2 = ax.plot(x, y_2, "--", c=ORANGE, label=r"$\loss(\h_{\S'}, (\x, \y))$")

i_max = np.argmax(abs(y_1-y_2))

ax.annotate("", xy=(x[i_max], np.minimum(y_1[i_max], y_2[i_max])),
            xytext=(x[i_max], np.maximum(y_1[i_max], y_2[i_max])),
            arrowprops=dict(arrowstyle="<->", color=BLACK))
ax.text(x[i_max]+0.1,
        (np.minimum(y_1[i_max], y_2[i_max])
         + np.maximum(y_1[i_max], y_2[i_max]))/2.0, r"$\le\stab$")

ax.set_xlabel(r"Learning samples $\S$ and $\S'$ \st $\Delta(\S,\S'){=}1$"
              + r" / Example $(\x, \y) \in \X\times\Y$")
ax.set_ylabel(r"Loss value")

ax.legend(loc='upper center',
          bbox_to_anchor=(0.5, 1.22),
          frameon=False, ncol=3)

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(left=False, labelleft=False,
               labelbottom=False, bottom=False)


os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/stability.pdf", bbox_inches="tight")
