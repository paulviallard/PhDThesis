import logging
import os
import sys
import numpy as np
import random
import torch

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


sys.path.append("../chap-5-mv-sto/sourcecode/")
from voter.stump import DecisionStumpMV

from learner.bound_joint_learner import BoundJointLearner
from learner.bound_risk_learner import BoundRiskLearner
from learner.bound_rand_learner import BoundRandLearner
from learner.c_bound_seeger_learner import CBoundSeegerLearner
from learner.stochastic_majority_vote_learner import (
    StochasticMajorityVoteLearner)
from core.modules import Modules
from core.writer import Writer

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/slides_header_standalone.tex", "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)

plt.rcParams.update({
    "font.size": 8,
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
    "scatter_bg_cmap", [BLUE, RED])

###############################################################################

seed = 42

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------- #

logging.basicConfig(level=logging.INFO)
#  logging.getLogger().disabled = True
logging.StreamHandler.terminator = ""

x_train, y_train = make_moons(n_samples=1000, noise=0.02, random_state=0)
x_train = (x_train - x_train.min(axis=0))/(
    x_train.max(axis=0) - x_train.min(axis=0))
y_train = 2*y_train-1
y_train = np.expand_dims(y_train, 1)

majority_vote = DecisionStumpMV(
    x_train, y_train, nb_per_attribute=32, complemented=True)

epoch = 1000
delta = 0.05

plot_list = ["risk", "bound", "ours"]

learner_all_dict = {
    "bound": {
        "bound-joint": None,
    },
    "risk": {
        "bound-joint": None,
    },
    "ours": {
        "bound-sto": None
    }
}


for plot in plot_list:

    fig = plt.figure(figsize=(10, 2*3))
    fig.subplots_adjust(wspace=0.31, hspace=0.5)
    gs = fig.add_gridspec(2, 2*3)

    if(plot == "risk"):
        # To remove the bound
        m = 10**10
        learner_dict = learner_all_dict["risk"]
        fig = plt.figure(figsize=(3, 2))
        fig.subplots_adjust(wspace=0.31, hspace=0.5)
        gs = fig.add_gridspec(1, 1)
    elif(plot == "bound"):
        m = len(x_train)
        learner_dict = learner_all_dict["bound"]
        fig = plt.figure(figsize=(3, 2))
        fig.subplots_adjust(wspace=0.31, hspace=0.5)
        gs = fig.add_gridspec(1, 1)
    else:
        m = len(x_train)
        learner_dict = learner_all_dict["ours"]
        fig = plt.figure(figsize=(3, 2))
        fig.subplots_adjust(wspace=0.31, hspace=0.5)
        gs = fig.add_gridspec(1, 1)

    for i in range(len(learner_dict.keys())):

        learner = list(learner_dict.keys())[i]
        if(i == 0):
            ax = plt.subplot(gs[0, 0:2])
        elif(i == 1):
            ax = plt.subplot(gs[0, 2:4])
        elif(i == 2):
            ax = plt.subplot(gs[0, 4:6])
        elif(i == 3):
            ax = plt.subplot(gs[1, 1:3])
        elif(i == 4):
            ax = plt.subplot(gs[1, 3:5])

        ax.set_title(learner_dict[learner])

        # We learn the weights of the MV with a PAC-Bayesian bound
        if(learner == "c-bound-seeger"):
            learner_ = CBoundSeegerLearner(
                majority_vote, epoch=epoch, m=m, delta=delta)
        elif(learner == "bound-risk"):
            learner_ = BoundRiskLearner(
                majority_vote, epoch=epoch, m=m, delta=delta)
        elif(learner == "bound-joint"):
            learner_ = BoundJointLearner(
                majority_vote, epoch=epoch, m=m, delta=delta)
        elif(learner == "bound-sto"):
            learner_ = StochasticMajorityVoteLearner(
                majority_vote, epoch, risk="exact", m=m, delta=delta)
        elif(learner == "bound-rand"):
            learner_ = BoundRandLearner(
                majority_vote, epoch, m=m, rand_n=100)

        # ------------------------------------------------------------------- #

        learner_.fit(x=x_train, y=y_train)
        model = learner_.mv_diff

        if(learner == "c-bound-seeger"):
            bound_ = Modules("CBoundLacasse", model, m=m, delta=delta).fit
        elif(learner == "bound-risk"):
            bound_ = Modules("BoundRisk", model, m=m, delta=delta).fit
        elif(learner == "bound-joint"):
            bound_ = Modules("BoundJoint", model, m=m, delta=delta).fit
        elif(learner == "bound-sto"):
            bound_ = Modules("BoundSto", model, m=m, delta=delta).fit
        elif(learner == "bound-rand"):
            bound_ = Modules(
                "BoundRand", model, m=m, delta=delta, rand_n=100).fit
        b = float(bound_(x=x_train, y=y_train))

        # ------------------------------------------------------------------- #

        x_1, x_2 = np.meshgrid(
            np.arange(-0.05, 1.05, 0.005),
            np.arange(-0.05, 1.05, 0.005))
        x_ = np.concatenate(
            (x_1.reshape((-1, 1)), x_2.reshape((-1, 1))), axis=1)
        y_ = model.predict(x_).reshape(x_1.shape)

        ax.contourf(x_1, x_2, y_, cmap=scatter_bg_cmap, alpha=0.6)

        ax.scatter(
            x_train[:, 0][y_train[:, 0] == -1],
            x_train[:, 1][y_train[:, 0] == -1],
            c=BLUE, marker="o", edgecolor=BLACK, s=20)
        ax.scatter(
            x_train[:, 0][y_train[:, 0] == 1],
            x_train[:, 1][y_train[:, 0] == 1],
            c=RED, marker="^", edgecolor=BLACK, s=20)

        ax.tick_params(left=False, labelleft=False,
                       labelbottom=False, bottom=False)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        ax.text(0.0, 0.0, r"Bound value: ${:.2f}$".format(b))

    os.makedirs("figures/", exist_ok=True)
    if(plot == "risk"):
        fig.savefig("figures/moons_risk.pdf", bbox_inches="tight")
    elif(plot == "bound"):
        fig.savefig("figures/moons_bound.pdf", bbox_inches="tight")
    else:
        fig.savefig("figures/moons_ours.pdf", bbox_inches="tight")
