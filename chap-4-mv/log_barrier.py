import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings

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


# Log-barrier extension
def log_barrier(x, t=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        term_1 = -(1.0/t)*np.log(-x)
    term_1[np.isnan(term_1)] = 0.0
    term_2 = t*x - (1.0/t)*np.log(1/(t**2.0))+(1/t)
    log_barrier = (
        term_1*(x <= -1.0/(t**2.0)).astype(float)
        + term_2*(x > -1.0/(t**2.0)).astype(float))
    return log_barrier

###############################################################################


fig, ax = plt.subplots(1, 1, figsize=(7, 3))
fig.subplots_adjust(wspace=0.14, hspace=0)

x = np.arange(-1, 0.1, 0.001)

# Log-barrier extension
ax.plot(x, log_barrier(x, t=10), c=BLACK,
        label=r"Log-barrier Extension $\logbar_{\lambda}()$")
ax.plot(x, log_barrier(x, t=10), c=blue_cmap(0))
ax.plot(x, log_barrier(x, t=20), c=blue_cmap(0.5))
ax.plot(x, log_barrier(x, t=100), c=blue_cmap(1.0))

ax.set_ylim(-0.1, 1.0)
ax.set_xlim(-1.0, 0.1)

# Barrier function
ax.plot(np.array([-1, 0]), np.array([0, 0]),
        "--", c=BLACK, label=r"Barrier $\logbar()$")
ax.plot(np.array([0, 0]), np.array([0, 1]), "--", c=BLACK)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.25),
    frameon=False, ncol=2, fontsize=17)

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/log_barrier.pdf", bbox_inches="tight")
