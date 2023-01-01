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

    prior = 0.5
    lr_list = [0.000001, 0.0001]
    data_list = ["mnist", "fashion", "cifar10"]
    var = 0.001

    bound_ours_mean = []
    bound_riv_mean = []
    bound_cat_mean = []
    bound_bla_mean = []
    test_risk_ours_mean = []
    test_risk_riv_mean = []
    test_risk_cat_mean = []
    test_risk_bla_mean = []
    emp_risk_ours_mean = []
    emp_risk_riv_mean = []
    emp_risk_cat_mean = []
    emp_risk_bla_mean = []
    div_ours_mean = []
    div_riv_mean = []
    div_cat_mean = []
    div_bla_mean = []
    div_ours_std = []
    div_riv_std = []
    div_cat_std = []
    div_bla_std = []

    bound_kl_sample = []
    div_kl = []

    for i in range(len(lr_list)):
        bound_ours_mean.append([])
        bound_riv_mean.append([])
        bound_cat_mean.append([])
        bound_bla_mean.append([])
        test_risk_ours_mean.append([])
        test_risk_riv_mean.append([])
        test_risk_cat_mean.append([])
        test_risk_bla_mean.append([])
        emp_risk_ours_mean.append([])
        emp_risk_riv_mean.append([])
        emp_risk_cat_mean.append([])
        emp_risk_bla_mean.append([])
        div_ours_mean.append([])
        div_riv_mean.append([])
        div_cat_mean.append([])
        div_bla_mean.append([])
        div_riv_std.append([])
        div_cat_std.append([])
        div_bla_std.append([])
        bound_kl_sample.append([])
        div_kl.append([])

        for j in range(len(data_list)):
            bound_ours_mean[i].append([])
            bound_riv_mean[i].append([])
            bound_cat_mean[i].append([])
            bound_bla_mean[i].append([])
            test_risk_ours_mean[i].append([])
            test_risk_riv_mean[i].append([])
            test_risk_cat_mean[i].append([])
            test_risk_bla_mean[i].append([])
            emp_risk_ours_mean[i].append(None)
            emp_risk_riv_mean[i].append(None)
            emp_risk_cat_mean[i].append(None)
            emp_risk_bla_mean[i].append(None)
            div_ours_mean[i].append(None)
            div_riv_mean[i].append(None)
            div_cat_mean[i].append(None)
            div_bla_mean[i].append(None)
            div_riv_std[i].append(None)
            div_cat_std[i].append(None)
            div_bla_std[i].append(None)

            bound_kl_sample[i].append(None)
            div_kl[i].append(None)

    for lr, data in itertools.product(*[lr_list, data_list]):

        i_data = data_list.index(data)
        i_lr = lr_list.index(lr)

        d_ours = data_.get(
            "bound-ours-mean", "test-risk-mean", "post-risk-mean",
            "div-renyi", "div-kl", "bound-kl-sample",
            data=data, prior=prior, var=var, post_lr=lr, bound="ours")

        bound_ours_mean[i_lr][i_data].append(
            d_ours["bound-ours-mean"].to_numpy()[0])
        test_risk_ours_mean[i_lr][i_data].append(
            d_ours["test-risk-mean"].to_numpy()[0])
        emp_risk_ours_mean[i_lr][i_data] = (
            d_ours["post-risk-mean"].to_numpy()[0])
        div_ours_mean[i_lr][i_data] = d_ours["div-renyi"].to_numpy()[0]
        bound_kl_sample[i_lr][i_data] = d_ours["bound-kl-sample"].to_numpy()[0]
        div_kl[i_lr][i_data] = d_ours["div-kl"].to_numpy()[0]

    data_list = [r"MNIST", r"FashionMNIST", r"CIFAR-10"]
    lr_list = [r'$\lr=10^{-6}$', r'$\lr=10^{-4}$']

    # https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
    # A4 size = (8.27, 11.69)
    fig, axs = plt.subplots(len(lr_list), len(data_list),
                            figsize=((8.27, 0.6*11.69)))

    plt.setp(axs[:, 0].flat, ylabel=r'Bound / Test Risk $R_{\mathcal{T}}(h)$')
    # Padding in points for annotate
    pad = 5

    hatch = [r"/", r"|", r"\\", r"-"]

    for i in range(len(lr_list)):
        for j in range(len(data_list)):

            bar_1 = axs[i, j].bar(
                0, bound_ours_mean[i][j], hatch=hatch[0], fill=False)
            bar_2 = axs[i, j].bar(1, bound_kl_sample[i][j], fill=False)

            rect_2 = list(enumerate(bar_2))[0][1]
            axs[i, j].text(
                rect_2.get_x() + rect_2.get_width()/2.,
                0.03*rect_2.get_height(),
                (r"$R_{{\mathcal{{S}}}}(h)$: {:.3f}, Rényi div.: {:.1f},"
                 + " KL div.: {:.1f}").format(
                     emp_risk_ours_mean[i][j], div_ours_mean[i][j],
                     div_kl[i][j]),
                ha='center', va='bottom', rotation=90, fontsize=9)

            axs[i, j].bar(
                0, test_risk_ours_mean[i][j], hatch=hatch[0],
                color=BLACK, alpha=0.6)

            y_min, y_max = axs[i, j].get_ylim()
            axs[i, j].set_xticks([0, 1])
            axs[i, j].set_xticklabels(
                [r"\algoours", r"\algostoNN"], fontsize=8)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               frameon=False, ncol=len(labels), fontsize=8)

    for ax, data in zip(axs[0, :], data_list):
        ax.annotate(data, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, lr in zip(axs[:, 0], lr_list):
        ax.annotate(lr, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    fig.tight_layout(rect=[0, 0.0, 1, 1])
    os.makedirs("figures/", exist_ok=True)
    plt.savefig("figures/plot_2_prior_"+str(prior)+"_var_"+str(var)+".pdf",
                bbox_inches="tight")
