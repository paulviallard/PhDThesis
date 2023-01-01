import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neural_network import MLPClassifier
import torch

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

# Good seed: [5, 8, 17]
# learn the NN with sklearn
NN = MLPClassifier(hidden_layer_sizes=(2, 2, 2, 2), verbose=True,
                   max_iter=100000, tol=1e-100, activation="tanh",
                   random_state=8)
NN.fit(x, y)


# Get the hidden layers with PyTorch
def neural_network(x_, start=0):
    x_pred = torch.tensor(x_)
    x_pred_list = []
    for i in range(start, len(NN.coefs_)):
        weights = torch.permute(torch.tensor(NN.coefs_[i]), (1, 0))
        bias = torch.tensor(NN.intercepts_[i])
        x_pred = torch.nn.functional.linear(x_pred, weights, bias=bias)

        if(i != len(NN.coefs_)-1):
            x_pred = torch.tanh(x_pred)
        else:
            x_pred = torch.sign(x_pred)
            x_pred = (0.5*(1+x_pred))[:, 0]

        x_pred_list.append(x_pred.numpy())

    return x_pred.numpy(), x_pred_list

# --------------------------------------------------------------------------- #


x_1, x_2 = np.meshgrid(
    np.arange(-1.05, 1.05, 0.001),
    np.arange(-1.05, 1.05, 0.001))
x_ = np.concatenate((x_1.reshape((-1, 1)), x_2.reshape((-1, 1))), axis=1)

_, x_pred_list = neural_network(x)
fig, ax_list = plt.subplots(
    1, len(x_pred_list)-1, figsize=((len(x_pred_list)-1)*3, 3))

for i in range(len(x_pred_list)-1):
    ax = ax_list[i]
    ax.tick_params(
        left=False, labelleft=False, bottom=False, labelbottom=False)

    y_, _ = neural_network(x_, start=i+1)
    y_ = y_.reshape(x_1.shape)
    ax.contourf(x_1, x_2, y_, cmap=scatter_bg_cmap, alpha=0.8)

    x_pred = x_pred_list[i]
    ax.scatter(
        x_pred[:, 0][y == 0], x_pred[:, 1][y == 0], c=BLUE,
        marker="o", edgecolor=BLACK, s=30)
    ax.scatter(
        x_pred[:, 0][y == 1], x_pred[:, 1][y == 1], c=RED,
        marker="^", edgecolor=BLACK, s=30)
    ax.set_xlabel(r"$({})$".format(i+1))

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/neural_network_full.pdf",
            bbox_inches="tight")

# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 1, figsize=(3, 3))

x_1, x_2 = np.meshgrid(
    np.arange(-0.05, 1.05, 0.001),
    np.arange(-0.05, 1.05, 0.001))
x_ = np.concatenate((x_1.reshape((-1, 1)), x_2.reshape((-1, 1))), axis=1)
y_, _ = neural_network(x_)
y_ = y_.reshape(x_1.shape)
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
fig.savefig("figures/neural_network.pdf",
            bbox_inches="tight")
