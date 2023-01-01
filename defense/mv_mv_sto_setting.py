import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/slides_header_standalone.tex", "r")
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


def gaussian(x, mu=0, sigma=1):
    pdf = (1/sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2.0)
    pdf = pdf/np.sum(pdf)
    return pdf


def gaussian_mixture(x, mu_list=[0], sigma_list=[1], weight_list=[1.0]):
    mu_list = np.array(mu_list)
    mu_list = np.expand_dims(mu_list, 0)
    sigma_list = np.array(sigma_list)
    sigma_list = np.expand_dims(sigma_list, 0)
    x = np.expand_dims(x, 1)
    weight_list = np.array(weight_list)

    pdf = gaussian(x, mu=mu_list, sigma=sigma_list)
    pdf = pdf@weight_list
    pdf = pdf/np.sum(pdf)
    return pdf


def gibbs(x, f_alpha=None):
    if(f_alpha is None):
        f_alpha = np.zeros(x.shape)
    pdf = np.exp(-f_alpha)/np.sum(np.exp(-f_alpha))
    return pdf


###############################################################################

fig, ax = plt.subplots(1, 1, figsize=(2, 1))

ax.barh([0, 1, 2], [0.4, 0.1, 0.5],
        align='center', color=ORANGE, alpha=0.8)
ax.barh([0, 1, 2], [0.4, 0.1, 0.5], fill=None, edgecolor=ORANGE, linewidth=1.0)

ax.set_xlim(0.0, 0.5)
ax.set_yticks([0, 1, 2], labels=[r"$\h_1$", r"$\h_2$", r"$\h_3$"])

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.spines.bottom.set_visible(False)
ax.tick_params(bottom=False, labelbottom=False)

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/sto_mv_1.pdf", bbox_inches="tight")

# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 1, figsize=(2, 1))

ax.barh([0, 1, 2], [0.3, 0.3, 0.4],
        align='center', color=ORANGE, alpha=0.8)
ax.barh([0, 1, 2], [0.3, 0.3, 0.4], fill=None, edgecolor=ORANGE, linewidth=1.0)

ax.set_xlim(0.0, 0.5)
ax.set_yticks([0, 1, 2], labels=[r"$\h_1$", r"$\h_2$", r"$\h_3$"])

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.spines.bottom.set_visible(False)
ax.tick_params(bottom=False, labelbottom=False)

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/sto_mv_2.pdf", bbox_inches="tight")

# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 1, figsize=(2, 1))

ax.barh([0, 1, 2], [0.1, 0.45, 0.45],
        align='center', color=ORANGE, alpha=0.8)
ax.barh([0, 1, 2], [0.1, 0.45, 0.45],
        fill=None, edgecolor=ORANGE, linewidth=1.0)

ax.set_xlim(0.0, 0.5)
ax.set_yticks([0, 1, 2], labels=[r"$\h_1$", r"$\h_2$", r"$\h_3$"])

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.spines.bottom.set_visible(False)
ax.tick_params(bottom=False, labelbottom=False)

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/sto_mv_3.pdf", bbox_inches="tight")

# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 1, figsize=(2, 1))

ax.barh([0], [0.3],
        align='center', color=ORANGE, alpha=0.8)
ax.barh([0], [0.3],
        fill=None, edgecolor=ORANGE, linewidth=1.0)

ax.set_xlim(0.0, 0.5)
ax.set_yticks([0], labels=[r"$\MV_{\Q_1}$"])

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.spines.bottom.set_visible(False)
ax.spines.left.set_linewidth(3)
ax.tick_params(width=3)
ax.tick_params(bottom=False, labelbottom=False)

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/sto_mv_weight_1.pdf", bbox_inches="tight")

# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 1, figsize=(2, 1))

ax.barh([0], [0.2],
        align='center', color=ORANGE, alpha=0.8)
ax.barh([0], [0.2],
        fill=None, edgecolor=ORANGE, linewidth=1.0)

ax.set_xlim(0.0, 0.5)
ax.set_yticks([0], labels=[r"$\MV_{\Q_2}$"])

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.spines.bottom.set_visible(False)
ax.spines.left.set_linewidth(3)
ax.tick_params(width=3)
ax.tick_params(bottom=False, labelbottom=False)

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/sto_mv_weight_2.pdf", bbox_inches="tight")

# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 1, figsize=(2, 1))

ax.barh([0], [0.5],
        align='center', color=ORANGE, alpha=0.8)
ax.barh([0], [0.5],
        fill=None, edgecolor=ORANGE, linewidth=1.0)

ax.set_xlim(0.0, 0.5)
ax.set_yticks([0], labels=[r"$\MV_{\Q_3}$"])

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.spines.bottom.set_visible(False)
ax.spines.left.set_linewidth(3)
ax.tick_params(width=3)
ax.tick_params(bottom=False, labelbottom=False)

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/sto_mv_weight_3.pdf", bbox_inches="tight")
