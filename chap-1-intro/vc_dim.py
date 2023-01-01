import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from numpy.random import default_rng
from matplotlib.colors import LinearSegmentedColormap

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

seed_x = 2
seed_y = 1
rng_x = default_rng(seed=seed_x)
rng_y = default_rng(seed=seed_y)

size = [3, 3, 3, 4]
x = rng_x.random((sum(size), 2))
y = rng_y.binomial(1, 0.5, (sum(size),))

# --------------------------------------------------------------------------- #


def get_x_y(i):
    j_begin = sum(size[:i])
    j_end = j_begin+size[i]
    x_ = x[j_begin:j_end]
    y_ = y[j_begin:j_end]
    x_ = (x_-np.min(x_, axis=0))/(np.max(x_, axis=0)-np.min(x_, axis=0))
    return x_, y_


def fit(x_, i):
    x_train, y_train = get_x_y(i)
    learner = SVC(kernel="linear", C=10**2)
    learner.fit(x_train, y_train)
    return learner.predict(x_).astype(float)

# --------------------------------------------------------------------------- #


fig, ax_list = plt.subplots(1, 4, figsize=(10, 2))

for i in range(len(ax_list)):
    ax = ax_list[i]

    x_1, x_2 = np.meshgrid(
        np.arange(-0.5, 1.5, 0.005),
        np.arange(-0.5, 1.5, 0.005))
    x_ = np.concatenate((x_1.reshape((-1, 1)), x_2.reshape((-1, 1))), axis=1)
    y_ = fit(x_, i).reshape(x_1.shape)
    ax.contourf(x_1, x_2, y_, cmap=scatter_bg_cmap, alpha=0.8)

    x_, y_ = get_x_y(i)
    ax.scatter(
        x_[:, 0][y_ == 0], x_[:, 1][y_ == 0], c=BLUE,
        marker="o", edgecolor=BLACK, s=20)
    ax.scatter(
        x_[:, 0][y_ == 1], x_[:, 1][y_ == 1], c=RED,
        marker="^", edgecolor=BLACK, s=20)

    ax.tick_params(left=False, labelleft=False,
                   labelbottom=False, bottom=False)
    ax.set_xlabel(r"{\it ("+str(["a", "b", "c", "d"][i])+r")}")

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/vc_dim.pdf", bbox_inches="tight")
