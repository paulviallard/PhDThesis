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

epsilon = 0.1

###############################################################################

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
fig.subplots_adjust(wspace=0.14, hspace=0)

p = np.arange(0.0001, 1-0.0001, 0.0001)

# Plot 1
ax[0].plot(p, kl(0.0, p), label=r"$\kl(\EE_{\h\sim\Q}\Risk_{\dS}(\h)\|p)$")
ax[0].plot(p, np.ones(p.shape)*epsilon, "--", c=RED,
           label=r"Bound value $\threshold$")

p_max = kl_inv(0.0, epsilon, "MAX")

ax[0].scatter(p_max, epsilon, zorder=2, c=BLACK, marker="s")

ax[0].plot([p_max, p_max], [0, epsilon], ":", c=BLACK)

ax[0].set_xlim([0.0, p_max+0.02])
ax[0].set_ylim([0.0, epsilon+0.02])


ax[0].set_xticks(
    [0.0,  0.02, 0.04, 0.06, 0.08, p_max, 0.11],
    [r"$\Risk_{\dS}(\h)$", r"$0.02$", r"$0.04$", r"$0.06$", r"$0.08$",
     r"$\klmax$", r"$0.11$"])

ax[0].yaxis.set_major_formatter(plt.FormatStrFormatter(r"$%.2f$"))

ax[0].set_xlabel(r"$p\in(0, 1)$")

ax[0].legend(
    loc='upper center',
    bbox_to_anchor=(1.0, 1.32),
    frameon=False, ncol=3, fontsize=20)

# Plot 2
ax[1].plot(p, kl(0.2, p))
ax[1].plot(p, np.ones(p.shape)*epsilon, "--", c=RED)

p_min = kl_inv(0.2, epsilon, "MIN")
p_max = kl_inv(0.2, epsilon, "MAX")

ax[1].scatter(p_min, epsilon, zorder=2, c=BLACK)
ax[1].scatter(p_max, epsilon, zorder=2, c=BLACK, marker="s")

ax[1].plot([p_max, p_max], [0, epsilon], ":", c=BLACK)
ax[1].plot([p_min, p_min], [0, epsilon], ":", c=BLACK)

ax[1].set_ylim([0.0, epsilon+0.02])
ax[1].set_xlim([p_min-0.05, p_max+0.05])

ax[1].set_xticks(
    [0.02, p_min, 0.12, 0.2, 0.27, 0.35, p_max, 0.46],
    [r"$0.02$", r"$\klmin$", r"$0.12$",
     r"$\Risk_{\dS}(\h)$",
     r"$0.27$", r"$0.35$", r"$\klmax$", r"$0.46$"])
ax[1].set_yticks([], [])

ax[1].set_xlabel(r"$p\in(0, 1)$")

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/invert_kl_optim.pdf", bbox_inches="tight")
