import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma, digamma

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

###############################################################################

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
fig.subplots_adjust(wspace=0.14, hspace=0)

x = np.arange(0, 5.1, 0.01)

ax.plot(
    x, np.log(gamma(x)), label=r'$\ln(\Gamma(\alpha))$', c=ORANGE,
    linewidth=2, linestyle="dashed")
ax.plot(x, digamma(x), label=r'$\psi(\alpha)$', c=BLUE,
        linewidth=2, linestyle="-")
ax.set_xlim(0.0, 5.0)
ax.set_ylim(-10.0, 4.0)
ax.set_xlabel(r"$\alpha$")

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.25),
    frameon=False, ncol=2, fontsize=17)


os.makedirs("figures/", exist_ok=True)
plt.savefig("figures/digamma.pdf", bbox_inches="tight")
