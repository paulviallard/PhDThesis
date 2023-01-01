import os.path
from os import makedirs
import numpy as np
import matplotlib.pyplot as plt

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/header_standalone.tex", "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)

plt.rcParams.update({
    "font.size": 15,
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


def zero_one_loss(x):
    loss = (x <= 0).astype(float)
    return loss


def first_order_loss(x):
    loss = (1.0-x)
    return loss


def second_order_loss(x):
    loss = (1.0-x)**2.0

    return loss


###############################################################################


fig, ax = plt.subplots(1, 1, figsize=(10, 3))

ax.set_xlim(-1.0, +1.0)
ax.set_ylim(-0.03, 4.05)

x = np.arange(-1, +1, 0.001)
l1 = ax.plot(x, zero_one_loss(x), linewidth=3, c=BLUE,
             label=r"01-loss $\lossZO(\h, (\x,\y))$")
l2 = ax.plot(x, first_order_loss(x), "--", linewidth=3, c=ORANGE,
             label=r"$1^{\text{st}}$ order loss $\lossFirst(\h, (\x,\y))$")
l3 = ax.plot(x, second_order_loss(x), ":", linewidth=3, c=MAGENTA,
             label=r"$2^{\text{nd}}$ order loss $\lossSnd(\h, (\x,\y))$")
ax.set_xlabel(r"Margin $\Mah(\x, \y)$")
ax.set_ylabel(r"Loss value $\loss(\h, (\x,\y))$")

ax.legend(loc='upper center',
          bbox_to_anchor=(0.5, 1.22),
          frameon=False, ncol=3)


makedirs("figures/", exist_ok=True)
fig.savefig("figures/losses.pdf", bbox_inches="tight")
