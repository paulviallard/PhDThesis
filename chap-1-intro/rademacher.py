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

scatter_bg_cmap = LinearSegmentedColormap.from_list(
    "scatter_bg_cmap", [BLUE, WHITE, ORANGE])

###############################################################################

seed_x = 2
seed_y = 1
rng_x = default_rng(seed=seed_x)
rng_y = default_rng(seed=seed_y)

x = rng_x.random((3, 2))
y = np.array([0, 1, 2])
rad_list = np.array([[+1, +1, +1], [+1, -1, -1], [-1, +1, +1], [-1, -1, -1]])

# --------------------------------------------------------------------------- #


def fit(x_, i):
    rad = rad_list[i]
    y_ = np.array(y)
    if(len(np.unique(rad)) == 2):
        y_[rad == +1] = y_[rad == -1][0]
    if(len(np.unique(rad)) == 1 and rad[0] == +1):
        y_ = (y_+1) % 3

    if(len(np.unique(y_)) == 1):
        pred = (y_[0])*np.ones(x_.shape[0])
        pred[0] = 0
        pred[1] = 1
        pred[2] = 2
        return pred

    learner = SVC(kernel="linear", C=10**2)
    learner.fit(x, y_)
    pred = learner.predict(x_).astype(float)
    pred[0] = 0
    pred[1] = 1
    pred[2] = 2
    return pred

# --------------------------------------------------------------------------- #


fig, ax_list = plt.subplots(1, 4, figsize=(10, 2))

for i in range(len(ax_list)):
    ax = ax_list[i]
    rad = rad_list[i]

    x_1, x_2 = np.meshgrid(
        np.arange(-0.5, 1.5, 0.005),
        np.arange(-0.5, 1.5, 0.005))
    x_ = np.concatenate((x_1.reshape((-1, 1)), x_2.reshape((-1, 1))), axis=1)
    y_ = fit(x_, i).reshape(x_1.shape)
    ax.contourf(x_1, x_2, y_, cmap=scatter_bg_cmap, alpha=0.8)

    ax.scatter(
        x[:, 0][y == 0], x[:, 1][y == 0], c=BLUE,
        marker="o", edgecolor=BLACK, s=20)
    ax.scatter(
        x[:, 0][y == 1], x[:, 1][y == 1], c=RED,
        marker="^", edgecolor=BLACK, s=20)
    ax.scatter(
        x[:, 0][y == 2], x[:, 1][y == 2], c=ORANGE,
        marker="s", edgecolor=BLACK, s=20)

    for i in range(3):
        if(i == 0 or i == 2):
            coord = (x[i, 0]-0.3, x[i, 1]+0.1)
        else:
            coord = (x[i, 0]-0.3, x[i, 1]-0.2)
        if(rad[i] == 1):
            text = r"$\kappa_"+str(i+1)+"{=}{+}1$"
        else:
            text = r"$\kappa_"+str(i+1)+"{=}{-}1$"
        ax.annotate(text, coord)

    ax.tick_params(left=False, labelleft=False,
                   labelbottom=False, bottom=False)

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/rademacher.pdf", bbox_inches="tight")
