import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.svm import LinearSVC

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

x = np.array(
    [[0, 0],
     [0, 1],
     [1, 1],
     [1, 0]])
y = np.array(
    [0, +1, 0, +1])


def linear_classifier(x_, w_1, w_2, w_3):
    return (w_1*x_[:, 0]+w_2*x_[:, 1]+w_3 <= 0).astype(float)

# --------------------------------------------------------------------------- #


fig, ax = plt.subplots(1, 1, figsize=(5, 3))

x_1, x_2 = np.meshgrid(
    np.arange(-0.5, 1.5, 0.005),
    np.arange(-0.5, 1.5, 0.005))
x_ = np.concatenate((x_1.reshape((-1, 1)), x_2.reshape((-1, 1))), axis=1)
y_ = linear_classifier(x_, 1.0, 1.0, -1.5).reshape(x_1.shape)
ax.contourf(x_1, x_2, y_, cmap=scatter_bg_cmap, alpha=0.8)

ax.scatter(
    x[:, 0][y == 0], x[:, 1][y == 0], c=BLUE,
    marker="o", edgecolor=BLACK, s=20)
ax.scatter(
    x[:, 0][y == 1], x[:, 1][y == 1], c=RED,
    marker="^", edgecolor=BLACK, s=20)

ax.tick_params(
    left=False, labelleft=False, bottom=False, labelbottom=False)

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/xor.pdf",
            bbox_inches="tight")
