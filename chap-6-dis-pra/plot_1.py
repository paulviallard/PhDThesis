import os
import sys
import itertools
import matplotlib.pyplot as plt

sys.path.append("sourcecode")
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
    lr_list = [0.000001, 0.0001]
    data_list = ["mnist", "fashion", "cifar10"]
    var_list = [0.000001, 0.00001, 0.0001, 0.001]

    bound_ours_mean = []
    bound_riv_mean = []
    bound_cat_mean = []
    bound_bla_mean = []
    test_risk_ours_mean = []
    test_risk_riv_mean = []
    test_risk_cat_mean = []
    test_risk_bla_mean = []

    for i in range(len(var_list)*len(lr_list)):
        bound_ours_mean.append([])
        bound_riv_mean.append([])
        bound_cat_mean.append([])
        bound_bla_mean.append([])
        test_risk_ours_mean.append([])
        test_risk_riv_mean.append([])
        test_risk_cat_mean.append([])
        test_risk_bla_mean.append([])

        for j in range(len(data_list)):
            bound_ours_mean[i].append([])
            bound_riv_mean[i].append([])
            bound_cat_mean[i].append([])
            bound_bla_mean[i].append([])
            test_risk_ours_mean[i].append([])
            test_risk_riv_mean[i].append([])
            test_risk_cat_mean[i].append([])
            test_risk_bla_mean[i].append([])

    for prior, var, lr, data in itertools.product(
        *[prior_list, var_list, lr_list, data_list]
    ):

        d_ours = data_.get(
            "bound-ours-mean", "test-risk-mean",
            data=data, prior=prior, var=var, post_lr=lr, bound="ours")
        d_riv = data_.get(
            "bound-rivasplata-mean", "test-risk-mean",
            data=data, prior=prior, var=var, post_lr=lr, bound="rivasplata")
        d_cat = data_.get(
            "bound-catoni-mean", "test-risk-mean",
            data=data, prior=prior, var=var, post_lr=lr, bound="catoni")
        d_bla = data_.get(
            "bound-blanchard-mean", "test-risk-mean",
            data=data, prior=prior, var=var, post_lr=lr, bound="blanchard")

        i_data = data_list.index(data)
        i_var = var_list.index(var)
        i_lr = lr_list.index(lr)

        bound_ours_mean[i_var+(i_lr)*len(var_list)][i_data].append(
            d_ours["bound-ours-mean"].to_numpy()[0])
        bound_riv_mean[i_var+(i_lr)*len(var_list)][i_data].append(
            d_riv["bound-rivasplata-mean"].to_numpy()[0])
        bound_cat_mean[i_var+(i_lr)*len(var_list)][i_data].append(
            d_cat["bound-catoni-mean"].to_numpy()[0])
        bound_bla_mean[i_var+(i_lr)*len(var_list)][i_data].append(
            d_bla["bound-blanchard-mean"].to_numpy()[0])

        test_risk_ours_mean[i_var+(i_lr)*len(var_list)][i_data].append(
            d_ours["test-risk-mean"].to_numpy()[0])
        test_risk_riv_mean[i_var+(i_lr)*len(var_list)][i_data].append(
            d_riv["test-risk-mean"].to_numpy()[0])
        test_risk_cat_mean[i_var+(i_lr)*len(var_list)][i_data].append(
            d_cat["test-risk-mean"].to_numpy()[0])
        test_risk_bla_mean[i_var+(i_lr)*len(var_list)][i_data].append(
            d_bla["test-risk-mean"].to_numpy()[0])

    prior_list = [str(prior) for prior in prior_list]
    data_list = [r"MNIST", r"FashionMNIST", r"CIFAR-10"]
    var_list = [
        r'$\sigma^2=10^{-6}$', r'$\sigma^2=10^{-5}$',
        r'$\sigma^2=10^{-4}$', r'$\sigma^2=10^{-3}$']
    lr_list = [r'$\lr=10^{-6}$', r'$\lr=10^{-4}$']
    lr_list_ = [0.000001, 0.0001]

    for k in range(len(lr_list)):

        # https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
        # A4 size = (8.27, 11.69)
        fig, axs = plt.subplots(
            len(var_list), len(data_list), figsize=((8.27, 0.55*11.69)))
        plt.setp(axs[:, 0].flat, ylabel='Bound')
        plt.setp(axs[len(var_list)-1, :].flat, xlabel='Ratio')
        # Padding in points for annotate
        pad = 5

        for i in range(len(var_list)):
            for j in range(len(data_list)):
                axs[i, j].plot(
                    prior_list, bound_ours_mean[i+(k*len(var_list))][j],
                    '-o', c=BLACK, label=r"\algoours")
                axs[i, j].plot(
                    prior_list, bound_cat_mean[i+(k*len(var_list))][j],
                    '-s', c=BLUE, label=r"\algocatoni")
                axs[i, j].plot(
                    prior_list, bound_riv_mean[i+(k*len(var_list))][j],
                    '-*', c=RED, label=r"\algorivasplata")
                axs[i, j].plot(
                    prior_list, bound_bla_mean[i+(k*len(var_list))][j],
                    '-^', c=GREEN, label=r"\algoblanchard")

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   frameon=False, ncol=len(labels))

        for ax, data in zip(axs[0, :], data_list):
            ax.annotate(data, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
        for ax, lr in zip(axs[:, 0], var_list):
            ax.annotate(lr, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        os.makedirs("figures/", exist_ok=True)
        plt.savefig("figures/plot_1_lr_"+str(lr_list_[k])+".pdf",
                    bbox_inches="tight")
