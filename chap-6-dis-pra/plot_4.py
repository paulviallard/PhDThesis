import os
import sys
import itertools
import matplotlib.pyplot as plt

sys.path.append("sourcecode/")
from core.nd_data import NDData

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/header_standalone.tex", "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)

plt.rcParams.update({
    "font.size": 10,
    "text.usetex": True,
    "pgf.rcfonts": False,
    "text.latex.preamble": preamble,
    "pgf.preamble": preamble,
})

###############################################################################

BLACK = "#000000"
BLUE = "#0077BB"
CYAN = "#009988"
GREEN = "#009988"
ORANGE = "#EE7733"
RED = "#CC3311"
MAGENTA = "#EE3377"
GREY = "#BBBBBB"

###############################################################################

if __name__ == "__main__":

    data_ = NDData("sourcecode/exp.csv")

    prior_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    bound_ours_mean = []
    bound_riv_mean = []
    bound_cat_mean = []
    bound_bla_mean = []
    bound_prior_ours_mean = []
    bound_prior_riv_mean = []
    bound_prior_cat_mean = []
    bound_prior_bla_mean = []

    for i in range(2):
        bound_prior_ours_mean.append([])
        bound_prior_riv_mean.append([])
        bound_prior_cat_mean.append([])
        bound_prior_bla_mean.append([])
        bound_ours_mean.append([])
        bound_riv_mean.append([])
        bound_cat_mean.append([])
        bound_bla_mean.append([])
        for j in range(3):
            bound_prior_ours_mean[i].append([])
            bound_prior_riv_mean[i].append([])
            bound_prior_cat_mean[i].append([])
            bound_prior_bla_mean[i].append([])
            bound_ours_mean[i].append([])
            bound_riv_mean[i].append([])
            bound_cat_mean[i].append([])
            bound_bla_mean[i].append([])

    prior = 0.5
    var_list = [0.000001, 0.00001, 0.0001, 0.001]
    post_lr_list = [0.000001, 0.0001]
    data_list = ["mnist", "fashion", "cifar10"]

    for var, post_lr, data in itertools.product(
        *[var_list, post_lr_list, data_list]
    ):

        i_post_lr = post_lr_list.index(post_lr)
        i_var = var_list.index(var)
        i_data = data_list.index(data)

        d_ours = data_.get(
            "bound-ours-mean", "test-risk-mean",
            data=data, prior=prior, var=var, post_lr=post_lr, bound="ours")
        d_riv = data_.get(
            "bound-rivasplata-mean", "test-risk-mean",
            data=data, prior=prior, var=var, post_lr=post_lr,
            bound="rivasplata")
        d_cat = data_.get(
            "bound-catoni-mean", "test-risk-mean",
            data=data, prior=prior, var=var, post_lr=post_lr,
            bound="catoni")
        d_bla = data_.get(
            "bound-blanchard-mean", "test-risk-mean",
            data=data, prior=prior, var=var, post_lr=post_lr,
            bound="blanchard")
        d_prior = data_.get(
            "bound-ours-mean", "bound-rivasplata-mean", "bound-catoni-mean",
            "bound-blanchard-mean", "test-risk-mean",
            data=data, prior=prior, var=var, bound=None)

        bound_prior_ours_mean[i_post_lr][i_data].append(
            d_prior["bound-ours-mean"].to_numpy()[0])
        bound_prior_riv_mean[i_post_lr][i_data].append(
            d_prior["bound-rivasplata-mean"].to_numpy()[0])
        bound_prior_cat_mean[i_post_lr][i_data].append(
            d_prior["bound-catoni-mean"].to_numpy()[0])
        bound_prior_bla_mean[i_post_lr][i_data].append(
            d_prior["bound-blanchard-mean"].to_numpy()[0])

        bound_ours_mean[i_post_lr][i_data].append(
            d_ours["bound-ours-mean"].to_numpy()[0])
        bound_riv_mean[i_post_lr][i_data].append(
            d_riv["bound-rivasplata-mean"].to_numpy()[0])
        bound_cat_mean[i_post_lr][i_data].append(
            d_cat["bound-catoni-mean"].to_numpy()[0])
        bound_bla_mean[i_post_lr][i_data].append(
            d_bla["bound-blanchard-mean"].to_numpy()[0])

    var_list = [r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$']
    data_list = [r"MNIST", r"FashionMNIST", r"CIFAR-10"]
    post_lr_list = [r'$\lr=10^{-6}$', r'$\lr=10^{-4}$']

    # https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
    # A4 size = (8.27, 11.69)
    fig, axs = plt.subplots(2, 3, figsize=(8.27, 4.69))
    plt.setp(axs[:, 0].flat, ylabel=r'Bound')
    plt.setp(axs[1, :].flat, xlabel=r'Variance $\sigma^2$')
    # Padding in points for annotate
    pad = 5

    for i in range(2):
        for j in range(3):
            axs[i, j].plot(var_list, bound_ours_mean[i][j],
                           '-o', c=BLACK, label=r"$\algoours$")
            axs[i, j].plot(var_list, bound_cat_mean[i][j],
                           '-s', c=BLUE, label=r"$\algocatoni$")
            axs[i, j].plot(var_list, bound_riv_mean[i][j],
                           '-*', c=RED, label=r"\algorivasplata")
            axs[i, j].plot(var_list, bound_bla_mean[i][j],
                           '-^', c=GREEN, label=r"\algoblanchard")

            axs[i, j].plot(var_list, bound_prior_ours_mean[i][j],
                           '--o', c=BLACK)
            axs[i, j].plot(var_list, bound_prior_cat_mean[i][j],
                           '--s', c=BLUE)
            axs[i, j].plot(var_list, bound_prior_riv_mean[i][j],
                           '--*', c=RED)
            axs[i, j].plot(var_list, bound_prior_bla_mean[i][j],
                           '--^', c=GREEN)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               frameon=False, ncol=len(labels))

    for ax, data in zip(axs[0, :], data_list):
        ax.annotate(data, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, post_lr in zip(axs[:, 0], post_lr_list):
        ax.annotate(post_lr, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs("figures/", exist_ok=True)
    plt.savefig('figures/plot_4_prior_'+str(prior)+'.pdf',
                bbox_inches="tight")
