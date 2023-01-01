import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/header_standalone.tex", "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)

plt.rcParams.update({
    "font.size": 21,
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


def cartesian_to_simplex(x, y):
    x_ = x
    y_ = y
    return [x_, y_, 1-x_-y_]


def simplex_to_ternary(x, y, z):
    x_ = 0.5*(2.0*y+z)/(x+y+z)
    y_ = (np.sqrt(3.0)/2)*(z/(x+y+z))
    return [x_, y_]


###############################################################################


fig = plt.figure(figsize=(6.0, 5.0), dpi=100, constrained_layout=True)
gs = fig.add_gridspec(1, 1)

ax1 = fig.add_subplot(gs[0, 0:1])
ax1.set_title(" ")

# We plot the border
border_x, border_y = simplex_to_ternary(
    np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
ax1.plot(border_x, border_y, color="black", linewidth=1.2)
border_x, border_y = simplex_to_ternary(
    np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
ax1.plot(border_x, border_y, color="black", linewidth=1.2)
border_x, border_y = simplex_to_ternary(
    np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]))
ax1.plot(border_x, border_y, color="black", linewidth=1.2)

# We plot the decision boundaries
border_x, border_y = simplex_to_ternary(
    np.array([0.3333, 0.5]), np.array([0.3333, 0.5]), np.array([0.3333, 0.0]))
ax1.plot(border_x, border_y, color=BLUE, linewidth=1.2)
border_x, border_y = simplex_to_ternary(
    np.array([0.3333, 0.0]), np.array([0.3333, 0.5]), np.array([0.3333, 0.5]))
ax1.plot(border_x, border_y, color=BLUE, linewidth=1.2)
border_x, border_y = simplex_to_ternary(
    np.array([0.3333, 0.5]), np.array([0.3333, 0.0]), np.array([0.3333, 0.5]))
ax1.plot(border_x, border_y, color=BLUE, linewidth=1.2)

# We plot the decision boundaries area
border_x, border_y = simplex_to_ternary(
    np.array([0.3333, 0.0, 0.0, 0.5]),
    np.array([0.3333, 0.5, 1.0, 0.5]),
    np.array([0.3333, 0.5, 0.0, 0.0]))
poly = Polygon(np.concatenate(
    (np.expand_dims(border_x, 1), np.expand_dims(border_y, 1)), axis=1),
    closed=True, facecolor=BLUE, alpha=0.3)
ax1.add_patch(poly)

# We plot the decision boundaries area for the 1/2-margin
border_x, border_y = simplex_to_ternary(
    np.array([0.0, 0.0, 0.5]),
    np.array([0.5, 1.0, 0.5]),
    np.array([0.5, 0.0, 0.0]))
poly = Polygon(np.concatenate(
    (np.expand_dims(border_x, 1), np.expand_dims(border_y, 1)), axis=1),
    closed=True, hatch="//", edgecolor=RED, facecolor="None")
ax1.add_patch(poly)

# We plot the majority vote
border_x, border_y = simplex_to_ternary(
    np.array([0.15]), np.array([0.7]), np.array([0.15]))
ax1.scatter(border_x, border_y, zorder=10, color=BLACK)

# We plot the decision boundaries for the 1/2-margin
border_x, border_y = simplex_to_ternary(
    np.array([0.0, 0.5]), np.array([0.5, 0.5]), np.array([0.5, 0.0]))
ax1.plot(border_x, border_y, color=RED, linewidth=1.2)

# We plot the notations
plt.text(-0.55, -0.10, r"$\PP_{\h\sim\Q}\LB \h(\x)=3 \RB=1$")
plt.text(0.65, -0.10, r"$\PP_{\h\sim\Q}\LB \h(\x)=1 \RB=1$")
plt.text(0.12, 0.91, r"$\PP_{\h\sim\Q}\LB \h(\x)=2 \RB=1$")

plt.axis('off')
os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/margin.pdf", bbox_inches="tight")
