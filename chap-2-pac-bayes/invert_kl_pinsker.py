import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

scatter_cmap = LinearSegmentedColormap.from_list(
    "scatter_cmap", [BLUE, RED])
scatter_bg_cmap = LinearSegmentedColormap.from_list(
    "scatter_bg_cmap", [BLUE, WHITE])

blue_cmap = LinearSegmentedColormap.from_list(
    "blue_cmap", [BLUE, CYAN])
orange_cmap = LinearSegmentedColormap.from_list(
    "orange_cmap", [ORANGE, RED])

###############################################################################


def kl(q, p):
    if(0 < q and q < 1):
        return q*np.log(q/p) + (1.0-q)*np.log((1.0-q)/(1.0-p))
    elif(q == 0):
        return (1.0-q)*np.log((1.0-q)/(1.0-p))
    else:
        return q*np.log(q/p)


def kl_inv(q, epsilon, mode, tol=10**-9, nb_iter_max=1000):
    """
    Solve the optimization problem min{ p in [0, 1] | kl(q||p) <= epsilon }
    or max{ p in [0,1] | kl(q||p) <= epsilon } for q and epsilon fixed
    Parameters
    ----------
    q: float
        The parameter q of the kl divergence
    epsilon: float
        The upper bound on the kl divergence
    tol: float, optional
        The precision tolerance of the solution
    nb_iter_max: int, optinal
        The maximum number of iterations
    """
    assert mode == "MIN" or mode == "MAX"
    assert isinstance(q, float) and q >= 0 and q <= 1
    assert isinstance(epsilon, float) and epsilon > 0.0

    # We optimize the problem with the bisection method
    if(mode == "MAX"):
        p_max = 1.0
        p_min = q
    else:
        p_max = q
        p_min = 10.0**-9

    for _ in range(nb_iter_max):
        p = (p_min+p_max)/2.0

        if(kl(q, p) == epsilon or (p_max-p_min)/2.0 < tol):
            return p

        if(mode == "MAX" and kl(q, p) > epsilon):
            p_max = p
        elif(mode == "MAX" and kl(q, p) < epsilon):
            p_min = p
        elif(mode == "MIN" and kl(q, p) > epsilon):
            p_min = p
        elif(mode == "MIN" and kl(q, p) < epsilon):
            p_max = p

    return p

# --------------------------------------------------------------------------- #


epsilon_list = [0.001, 0.1, 0.3, 0.5, 0.9]
q_list = np.arange(0.0001, 1-0.0001, 0.01)


###############################################################################

fig, ax = plt.subplots(1, 2, figsize=(11, 3))
#  fig.subplots_adjust(wspace=0.14, hspace=0)
i = 0

for epsilon in epsilon_list:
    kl_inv_max_list = []
    kl_inv_min_list = []
    pinsker_max_list = []
    pinsker_min_list = []
    for q in list(q_list):
        kl_inv_max_list.append(kl_inv(q, epsilon, "MAX"))
        kl_inv_min_list.append(kl_inv(q, epsilon, "MIN"))
        pinsker_max_list.append(q + np.sqrt(0.5*epsilon))
        pinsker_min_list.append(q - np.sqrt(0.5*epsilon))

    # For the legend
    if(epsilon == epsilon_list[0]):
        ax[0].plot(q_list, kl_inv_min_list, c=BLACK,
                   label=r"$\klmin(\Risk_{\dS}(\h) | \threshold)$")
        ax[0].plot(q_list, pinsker_min_list, "--", c=BLACK,
                   label=r"$\Risk_{\dS}(\h){-}\sqrt{\tfrac{1}{2}\threshold}$")
        ax[1].plot(q_list, kl_inv_max_list, c=BLACK,
                   label=r"$\klmax(\Risk_{\dS}(\h) | \threshold)$")
        ax[1].plot(q_list, pinsker_max_list, "--", c=BLACK,
                   label=r"$\Risk_{\dS}(\h){+}\sqrt{\tfrac{1}{2}\threshold}$")

    ax[0].plot(q_list, kl_inv_min_list, c=blue_cmap(i/len(epsilon_list)))
    ax[1].plot(q_list, kl_inv_max_list, c=blue_cmap(i/len(epsilon_list)))

    ax[0].plot(
        q_list, pinsker_min_list, "--", c=orange_cmap(i/len(epsilon_list)))
    ax[1].plot(
        q_list, pinsker_max_list, "--", c=orange_cmap(i/len(epsilon_list)))

    i += 1

ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.30),
          frameon=False, ncol=3)
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.30),
          frameon=False, ncol=3)

ax[0].set_xlim(0.0, 1.0)
ax[0].set_ylim(0.0, 1.0)
ax[1].set_xlim(0.0, 1.0)
ax[1].set_ylim(0.0, 1.0)

ax[0].set_xlabel(r"Empirical risk $\Risk_{\dS}(\h)$")
ax[0].set_ylabel(r"Lower bound on $\Risk_{\D}(\h)$")
ax[1].set_xlabel(r"Empirical risk $\Risk_{\dS}(\h)$")
ax[1].set_ylabel(r"Upper bound on $\Risk_{\D}(\h)$")
# --------------------------------------------------------------------------- #

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/invert_kl_pinsker.pdf", bbox_inches="tight")
