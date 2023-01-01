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

###############################################################################


x = np.linspace(-4.0, 7.5, 1000)
dist = gaussian_mixture(
    x, mu_list=[0, 5], sigma_list=[1, 0.5], weight_list=[0.5, 0.5])

fig, ax = plt.subplots(1, 1, figsize=(6, 2))

ax.plot(x, dist, c=ORANGE)
ax.fill_between(x, dist, zorder=-1, fc=ORANGE, alpha=0.5)

ax.set_ylim(0.0)
ax.set_xlim(-3.0, 6.5)
ax.set_axis_off()

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/distribution.pdf", bbox_inches="tight")

ax.set_axis_on()
ax.set_xlabel(r"Example $(\x, \y)$")
ax.set_ylabel(r"$\D((\x, \y))$")

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(left=False, labelleft=False,
               labelbottom=False, bottom=False)


os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/distribution_all.pdf", bbox_inches="tight")
