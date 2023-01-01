import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import dirichlet

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/header_standalone.tex", "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)

plt.rcParams.update({
    "font.size": 12*3,
    "text.usetex": True,
    "pgf.rcfonts": False,
    "text.latex.preamble": preamble,
    "pgf.preamble": preamble,
})

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

def show_plot(alpha):

    scale = 200
    x, y = np.meshgrid(
        np.linspace(0.00001, 1.0, scale),
        np.linspace(0.00001, 1.0, scale))

    x_, y_, z_ = cartesian_to_simplex(x, y)

    x_ = x_.reshape(scale**2, 1)
    y_ = y_.reshape(scale**2, 1)
    z_ = z_.reshape(scale**2, 1)

    x__ = np.array(x_)
    x__[0.0 > z_] = 0.000001
    x__[z_ > 1.0] = 0.000001

    y__ = np.array(y_)
    y__[0.0 > z_] = 0.000001
    y__[z_ > 1.0] = 0.000001

    z__ = np.array(z_)
    z__[0.0 > z_] = 1.0-0.000002
    z__[z_ > 1.0] = 1.0-0.000002

    quantiles = np.array([x__, y__, z__]).squeeze()
    dir = dirichlet.pdf(quantiles, alpha)
    cond_1 = (0.0 > z_)
    cond_2 = (z_ > 1.0)
    dir = np.ma.array(dir, mask=cond_1+cond_2)

    x_ = x__.reshape(scale, scale)
    y_ = y__.reshape(scale, scale)
    z_ = z__.reshape(scale, scale)
    dir = dir.reshape(scale, scale)

    # ----------------------------------------------------------------------- #

    # We convert the simplex into the ternary plot coordinates
    x_, y_ = simplex_to_ternary(x_, y_, z_)

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

    # We plot the Dirichlet distribution
    cs = ax1.contourf(x_, y_, dir, 20, cmap=plt.get_cmap("YlOrBr"))
    ax1.contour(cs, colors='black', linewidths=0.01)

    # We plot the notations
    plt.text(-0.35, -0.10, r"$\Q(\h_1){=}1$")
    plt.text(0.85, -0.10, r"$\Q(\h_2){=}1$")
    plt.text(0.32, 0.91, r"$\Q(\h_3){=}1$")

    plt.axis('off')

    alpha = [str(int(a)).replace(".", "_") for a in list(alpha)]
    name_fig = "dirichlet_"+"_".join(alpha)+".pdf"
    os.makedirs("figures/", exist_ok=True)
    plt.savefig("figures/"+name_fig, bbox_inches="tight")


###############################################################################

alpha = np.array([1, 1, 1])
show_plot(alpha)
alpha = 2*np.array([1, 1, 1])
show_plot(alpha)
alpha = 10*np.array([1, 1, 1])
show_plot(alpha)

alpha = 10*np.array([0.5, 0.1, 0.4])
show_plot(alpha)
alpha = 20*np.array([0.5, 0.1, 0.4])
show_plot(alpha)
alpha = 50*np.array([0.5, 0.1, 0.4])
show_plot(alpha)
