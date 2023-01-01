import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/header_standalone.tex", "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)

plt.rcParams.update({
    "font.size": 25,
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

###############################################################################

fig, ax = plt.subplots(1, 3, figsize=(22, 7))
fig.subplots_adjust(wspace=0.31, hspace=0)

# With the margin ----------------------------------------------------------- #

m_1, m_2 = np.meshgrid(
    np.linspace(0.0, 1-0.001, 300), np.linspace(0.00001, 1-0.001, 300))
m_1_ = np.linspace(0.00001, 1-0.001, 400)
cb = 1.0-((m_1)**2.0)/(m_2)
cond_1 = (m_1 < 0)
cond_2 = (m_2 < m_1**2.0)
cb = np.ma.array(cb, mask=cond_1+cond_2)
gibbs = (1.0-m_1)
gibbs = np.ma.array(gibbs, mask=(gibbs > cb)+cond_1+cond_2)

ax[0].contourf(m_1, m_2, cb, 20)
ax[0].contourf(m_1, m_2, gibbs, 20, colors=BLACK, alpha=0.3)
ax[0].plot([0.0, 1.0], [0.0, 1.0], c=BLACK, linestyle="dashed")
ax[0].plot(m_1_, (m_1_**2.0), "black", linewidth=2)
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 1)
ax[0].set_xlabel(r"$\EE_{(\x,\y)\sim\Dp}\OmMaQ(\x,\y)$")
ax[0].set_ylabel(r"$\EE_{(\x,\y)\sim\Dp}\OmMaQ(\x,\y)^2$")

# With the risk/disagreement ------------------------------------------------ #

r, d = np.meshgrid(
    np.linspace(0.0, 0.5-0.001, 300), np.linspace(0.00001, 0.5-0.001, 300))
r_ = np.linspace(0.00001, 0.5-0.001, 400)
cb = 1.0-((1-2*r)**2.0)/(1-2*d)
cond_1 = (r > 1/2)
cond_2 = ((1-2*d) < (1-2*r)**2.0)

cb = np.ma.array(cb, mask=cond_1+cond_2)
gibbs = (2*r)
gibbs = np.ma.array(gibbs, mask=(gibbs > cb)+cond_1+cond_2)

ax[1].contourf(r, d, cb, 20)
ax[1].contourf(r, d, gibbs, 20, colors=BLACK, alpha=0.3)
ax[1].plot([0.0, 0.5], [0.0, 0.5], c=BLACK, linestyle="dashed")
ax[1].plot(r_, 0.5*(1-(1-2*r_)**2.0), "black", linewidth=2)

ax[1].set_xlim(0, 0.5)
ax[1].set_ylim(0, 0.5)
ax[1].set_xlabel(r"$r_{\Dp}(\Q)$")
ax[1].set_ylabel(r"$d_{\Dp}(\Q)$")

# With the joint error/disagreement ----------------------------------------- #

e, d = np.meshgrid(
    np.linspace(0.0, 0.5-0.001, 300), np.linspace(0.0, 0.5-0.001, 300))
e_ = np.linspace(0.0, 0.5-0.001, 400)
cb = (1.0-((1.0-(2.0*e+d))**2.0)/(1.0-2.0*d))
cond_1 = (2*e+d >= 1)
cond_2 = (d >= 2*(np.sqrt(e)-e))
cb = np.ma.array(cb, mask=cond_1+cond_2)

gibbs = (2.0*e+d)
gibbs = np.ma.array(gibbs, mask=(gibbs > cb)+cond_1+cond_2)

cs = ax[2].contourf(r, d, cb, 20)
ax[2].contourf(r, d, gibbs, 20, colors=BLACK, alpha=0.3)
ax[2].plot([0.0, 0.25], [0.0, 0.5], c=BLACK, linestyle="dashed")
ax[2].plot(e_, 2.0*(np.sqrt(np.minimum(0.25, e_))-e_), "black", linewidth=2)
ax[2].set_xlim(0, 0.5)
ax[2].set_ylim(0, 0.5)
ax[2].set_xlabel(r"$e_{\Dp}(\Q)$")
ax[2].set_ylabel(r"$d_{\Dp}(\Q)$")

# Legend -------------------------------------------------------------------- #

plt.colorbar(cs, ax=ax, orientation="vertical")

# --------------------------------------------------------------------------- #
os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/cbound.pdf", bbox_inches="tight")
