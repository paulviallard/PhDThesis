import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from matplotlib.colors import LinearSegmentedColormap

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/header_standalone.tex", "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)

plt.rcParams.update({
    "font.size": 20,
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

train_size = 50
# Good seed: [2, 4]
seed = 4

rng = default_rng(seed=seed)

x_1 = rng.normal((0.0, 0.0), 1.0, size=(train_size//2, 2))
y_1 = np.zeros(x_1.shape[0])
x_2 = rng.normal((2, 0.0), 1.0, size=(train_size//2, 2))
y_2 = np.ones(x_2.shape[0])
x = np.concatenate((x_1, x_2))
y = np.concatenate((y_1, y_2))
x = (np.max(x, axis=0)-x)/(np.max(x, axis=0)-np.min(x, axis=0))

# --------------------------------------------------------------------------- #


def decision_stump(x_):
    return (x_[:, 0] <= 0.54).astype(float)

# --------------------------------------------------------------------------- #


fig, ax = plt.subplots(1, 1, figsize=(3, 3))

x_1, x_2 = np.meshgrid(
    np.arange(-0.05, 1.05, 0.0005),
    np.arange(-0.05, 1.05, 0.0005))
x_ = np.concatenate((x_1.reshape((-1, 1)), x_2.reshape((-1, 1))), axis=1)
y_ = decision_stump(x_).reshape(x_1.shape)
ax.contourf(x_1, x_2, y_, cmap=scatter_bg_cmap, alpha=0.8)

ax.scatter(
    x[:, 0][y == 0], x[:, 1][y == 0], c=BLUE,
    marker="o", edgecolor=BLACK, s=20)
ax.scatter(
    x[:, 0][y == 1], x[:, 1][y == 1], c=RED,
    marker="^", edgecolor=BLACK, s=20)

ax.tick_params(left=False, labelleft=False)

ax.set_xticks([0.54])
ax.set_xticklabels([r"$0.54$"])
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/decision_stump.pdf", bbox_inches="tight")
