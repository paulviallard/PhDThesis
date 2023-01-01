import logging
import os
import numpy as np
import random
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/slides_header_standalone.tex", "r")
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

scatter_cmap = LinearSegmentedColormap.from_list(
    "scatter_cmap", [BLUE, RED])
scatter_bg_cmap = LinearSegmentedColormap.from_list(
    "scatter_bg_cmap", [BLUE, WHITE])


seed = 42

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###############################################################################


def himmelblau(x):
    x = 12.0*x-6.0
    return (x[:, 0]**2.0+x[:, 1]-11.0)**2.0 + (x[:, 0]+x[:, 1]**2.0 - 7.0)**2.0

###############################################################################


logging.basicConfig(level=logging.INFO)
logging.StreamHandler.terminator = ""


T = 10
lr = 0.0001

###############################################################################

x_1, x_2 = np.meshgrid(
    np.arange(-0.05, 1.05, 0.005), np.arange(-0.05, 1.05, 0.005))
x_ = np.concatenate((x_1.reshape((-1, 1)), x_2.reshape((-1, 1))), axis=1)
y_ = np.log(himmelblau(x_).reshape(x_1.shape))

fig, ax = plt.subplots(1, 1, figsize=(6, 2))

ax.contourf(x_1, x_2, y_, cmap=scatter_bg_cmap, alpha=0.8)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

ax.tick_params(left=False, labelleft=False,
               labelbottom=False, bottom=False)
ax.set_xlabel(r"$w_1$")
ax.set_ylabel(r"$w_2$")

x = torch.rand(1, 2, requires_grad=True)
x_list = x

for t in range(T):
    loss = himmelblau(x)
    loss.backward()
    grad = x.grad.clone().detach()
    x.grad.zero_()
    x.data = x.data-lr*(grad)+torch.rand(1, 2)*0.05
    x_list = torch.concat((x_list, x), axis=0)

x_list = x_list.detach().numpy()

ax.scatter(
    x_list[:, 0], x_list[:, 1],
    c=BLACK, marker="x", s=20)
ax.plot(x_list[:, 0], x_list[:, 1], c=RED)
ax.text(x_list[0, 0], x_list[0, 1]-0.13, r"$\h_1$")
ax.text(x_list[-1, 0], x_list[-1, 1]+0.05, r"$\h_T$")

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/perspectives_algo.pdf", bbox_inches="tight")
