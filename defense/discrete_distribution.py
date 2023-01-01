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


x = np.arange(0, 5)
emp_risk_list = np.array([0.04, 0.1, 0.05, 0.1, 0.02])

dist_dict = {
    "prior": gaussian_mixture(
        x, mu_list=[0], sigma_list=[10**10], weight_list=[1.0]),
    "post": gibbs(x, f_alpha=10*emp_risk_list),
}

y_max = np.max(np.concatenate([dist_dict["prior"], dist_dict["post"]]))


for dist in ["prior", "post"]:
    fig, ax = plt.subplots(1, 1, figsize=(6, 1))

    if(dist == "prior"):
        ax.bar(x, dist_dict["prior"], width=0.8, color=GREY)
        ax.bar(x, dist_dict["prior"], width=0.8, fill=None,
               edgecolor=BLACK, linewidth=1.0)
    elif(dist == "post"):
        ax.bar(x, dist_dict["post"], width=0.8, color=ORANGE, alpha=0.8)
        ax.bar(x, dist_dict["post"], width=0.8, fill=None,
               edgecolor=ORANGE, linewidth=1.0)

    ax.set_ylim(0.01, y_max)
    if(dist == "prior"):
        ax.set_ylabel(r"Prior $\P(\h)$")
    elif(dist == "post"):
        ax.set_ylabel(r"Posterior $\Q(\h)$")

    ax.set_xticks(x, [r"$\h_1$", r"$\h_2$", r"$\h_3$", r"$\h_4$", r"$\h_5$"])

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(left=False, labelleft=False,
                   bottom=False)

    os.makedirs("figures/", exist_ok=True)
    fig.savefig(f"figures/distribution_{dist}.pdf", bbox_inches="tight")
