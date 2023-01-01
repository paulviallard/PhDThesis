import os
import sys
import itertools
import matplotlib.pyplot as plt

sys.path.append("../chap-6-dis-pra/sourcecode")
from core.nd_data import NDData

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/slides_header_standalone.tex", "r")
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

    comp_type_list = ["all", "slides"]

    for comp_type in comp_type_list:

        data_ = NDData("../chap-6-dis-pra/sourcecode/exp.csv")

        prior = 0.5
        post_lr_list = [0.0001]
        data_list = ["mnist", "fashion", "cifar10"]
        var = 0.001
        ratio_more = 1.2

        bound_prior_ours_mean = []
        bound_prior_riv_mean = []
        bound_prior_cat_mean = []
        bound_prior_bla_mean = []
        bound_ours_mean = []
        bound_riv_mean = []
        bound_cat_mean = []
        bound_bla_mean = []
        test_risk_ours_mean = []
        test_risk_riv_mean = []
        test_risk_cat_mean = []
        test_risk_bla_mean = []
        test_risk_prior_mean = []
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

        for i in range(len(post_lr_list)):
            bound_prior_ours_mean.append([])
            bound_prior_riv_mean.append([])
            bound_prior_cat_mean.append([])
            bound_prior_bla_mean.append([])
            bound_ours_mean.append([])
            bound_riv_mean.append([])
            bound_cat_mean.append([])
            bound_bla_mean.append([])
            test_risk_ours_mean.append([])
            test_risk_riv_mean.append([])
            test_risk_cat_mean.append([])
            test_risk_bla_mean.append([])
            test_risk_prior_mean.append([])
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

            for j in range(len(data_list)):
                bound_prior_ours_mean[i].append([])
                bound_prior_riv_mean[i].append([])
                bound_prior_cat_mean[i].append([])
                bound_prior_bla_mean[i].append([])
                bound_ours_mean[i].append([])
                bound_riv_mean[i].append([])
                bound_cat_mean[i].append([])
                bound_bla_mean[i].append([])
                test_risk_ours_mean[i].append([])
                test_risk_riv_mean[i].append([])
                test_risk_cat_mean[i].append([])
                test_risk_bla_mean[i].append([])
                test_risk_prior_mean[i].append([])
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

        for post_lr, data in itertools.product(*[post_lr_list, data_list]):

            i_data = data_list.index(data)
            i_post_lr = post_lr_list.index(post_lr)

            d_ours = data_.get(
                "bound-ours-mean", "test-risk-mean", "post-risk-mean",
                "div-rivasplata-mean", "div-rivasplata-std", "div-renyi",
                data=data, prior=prior, var=var, post_lr=post_lr, bound="ours")
            d_riv = data_.get(
                "bound-rivasplata-mean", "test-risk-mean", "post-risk-mean",
                "div-rivasplata-mean", "div-rivasplata-std", "div-renyi",
                data=data, prior=prior, var=var, post_lr=post_lr,
                bound="rivasplata")

            d_cat = data_.get(
                "bound-catoni-mean", "test-risk-mean", "post-risk-mean",
                "div-rivasplata-mean", "div-rivasplata-std", "div-renyi",
                data=data, prior=prior, var=var, post_lr=post_lr,
                bound="catoni")
            d_bla = data_.get(
                "bound-blanchard-mean", "test-risk-mean", "post-risk-mean",
                "div-rivasplata-mean", "div-rivasplata-std", "div-renyi",
                data=data, prior=prior, var=var, post_lr=post_lr,
                bound="blanchard")

            d_prior = data_.get(
                "bound-ours-mean", "bound-rivasplata-mean",
                "bound-catoni-mean", "bound-blanchard-mean", "test-risk-mean",
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

            test_risk_ours_mean[i_post_lr][i_data].append(
                d_ours["test-risk-mean"].to_numpy()[0])
            test_risk_riv_mean[i_post_lr][i_data].append(
                d_riv["test-risk-mean"].to_numpy()[0])
            test_risk_cat_mean[i_post_lr][i_data].append(
                d_cat["test-risk-mean"].to_numpy()[0])
            test_risk_bla_mean[i_post_lr][i_data].append(
                d_bla["test-risk-mean"].to_numpy()[0])
            test_risk_prior_mean[i_post_lr][i_data].append(
                d_prior["test-risk-mean"].to_numpy()[0])

            emp_risk_ours_mean[i_post_lr][i_data] = (
                d_ours["post-risk-mean"].to_numpy()[0])
            emp_risk_riv_mean[i_post_lr][i_data] = (
                d_riv["post-risk-mean"].to_numpy()[0])
            emp_risk_cat_mean[i_post_lr][i_data] = (
                d_cat["post-risk-mean"].to_numpy()[0])
            emp_risk_bla_mean[i_post_lr][i_data] = (
                d_bla["post-risk-mean"].to_numpy()[0])

            div_ours_mean[i_post_lr][i_data] = (
                d_ours["div-renyi"].to_numpy()[0])
            div_riv_mean[i_post_lr][i_data] = (
                d_riv["div-rivasplata-mean"].to_numpy()[0])
            div_cat_mean[i_post_lr][i_data] = (
                d_cat["div-rivasplata-mean"].to_numpy()[0])
            div_bla_mean[i_post_lr][i_data] = (
                d_bla["div-rivasplata-mean"].to_numpy()[0])

            div_riv_std[i_post_lr][i_data] = (
                d_riv["div-rivasplata-std"].to_numpy()[0])
            div_cat_std[i_post_lr][i_data] = (
                d_cat["div-rivasplata-std"].to_numpy()[0])
            div_bla_std[i_post_lr][i_data] = (
                d_bla["div-rivasplata-std"].to_numpy()[0])

        data_list = ["MNIST", "FashionMNIST", "CIFAR-10"]
        post_lr_list = [r'$\lr=10^{-4}$']

        # https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
        if(comp_type == "all"):
            fig, ax_list = plt.subplots(len(post_lr_list), len(data_list),
                                        figsize=((8.27, 0.6*11.69*0.5)))
        elif(comp_type == "slides"):
            fig, ax_list = plt.subplots(len(post_lr_list), len(data_list),
                                        figsize=((0.8*8.27, 0.6*11.69*0.5)))

        if(comp_type == "all"):
            plt.setp(
                ax_list[0], ylabel=r'Bound --- Test Risk $\Risk_{\dT}(\h)$')
        # Padding in points for annotate
        pad = 5

        for j in range(len(data_list)):

            ax_list[j].bar(0, test_risk_ours_mean[i][j], hatch=r"\\",
                           color=RED)
            ax_list[j].bar(2, test_risk_riv_mean[i][j], hatch=r"\\",
                           color=RED)

            if(comp_type == "all"):
                ax_list[j].bar(1, test_risk_cat_mean[i][j], hatch=r"\\",
                               color=RED)
                ax_list[j].bar(3, test_risk_bla_mean[i][j], hatch=r"\\",
                               color=RED)

            if(comp_type == "all"):
                bar_1_prior = ax_list[j].bar(
                    0, bound_prior_ours_mean[i][j], fill=False,
                    linestyle="dashed", linewidth=1.2)
                bar_1_test_prior = ax_list[j].bar(
                    0, test_risk_prior_mean[i][j], fill=False,
                    linestyle="dotted", linewidth=1.4)
                bar_1 = ax_list[j].bar(0, bound_ours_mean[i][j],
                                       hatch=r"\\", fill=False)
                bar = bar_1
                if(bound_prior_ours_mean[i][j] > bound_ours_mean[i][j]):
                    bar = bar_1_prior
                ax_list[j].bar_label(bar, [r"{:.3f}, {:.1f}".format(
                    emp_risk_ours_mean[i][j], div_ours_mean[i][j]
                )], rotation=90, padding=5)

            elif(comp_type == "slides"):
                ax_list[j].bar(0, bound_ours_mean[i][j],
                               hatch=r"\\", fill=False)

            if(comp_type == "all"):
                bar_2_prior = ax_list[j].bar(
                    1, bound_prior_cat_mean[i][j], fill=False,
                    linestyle="dashed", linewidth=1.2)
                bar_2_test_prior = ax_list[j].bar(
                    1, test_risk_prior_mean[i][j], fill=False,
                    linestyle="dotted", linewidth=1.4)
                bar_2 = ax_list[j].bar(1, bound_cat_mean[i][j],
                                       hatch=r"\\", fill=False)
                ax_list[j].bar_label(
                    bar_2, [r"{:.3f}, {:.1f} $\pm$ {:.1f}".format(
                        emp_risk_cat_mean[i][j], div_cat_mean[i][j],
                        div_cat_std[i][j])], rotation=90, padding=5)

            if(comp_type == "all"):
                bar_3_prior = ax_list[j].bar(
                    2, bound_prior_riv_mean[i][j], fill=False,
                    linestyle="dashed", linewidth=1.2)
                bar_3_test_prior = ax_list[j].bar(
                    2, test_risk_prior_mean[i][j], fill=False,
                    linestyle="dotted", linewidth=1.4)
                bar_3 = ax_list[j].bar(
                    2, bound_riv_mean[i][j], hatch=r"\\", fill=False)
                ax_list[j].bar_label(
                    bar_3, [r"{:.3f}, {:.1f} $\pm$ {:.1f}".format(
                        emp_risk_riv_mean[i][j], div_riv_mean[i][j],
                        div_riv_std[i][j])], rotation=90, padding=5)

            elif(comp_type == "slides"):
                ax_list[j].bar(2, bound_riv_mean[i][j],
                               hatch=r"\\", fill=False)

            if(comp_type == "all"):
                bar_4_prior = ax_list[j].bar(
                    3, bound_prior_bla_mean[i][j], fill=False,
                    linestyle="dashed", linewidth=1.2)
                bar_4_test_prior = ax_list[j].bar(
                    3, test_risk_prior_mean[i][j], fill=False,
                    linestyle="dotted", linewidth=1.4)
                bar_4 = ax_list[j].bar(
                    3, bound_bla_mean[i][j], hatch=r"\\", fill=False)
                ax_list[j].bar_label(
                    bar_4, [r"{:.3f}, {:.1f} $\pm$ {:.1f}".format(
                        emp_risk_bla_mean[i][j], div_bla_mean[i][j],
                        div_bla_std[i][j])], rotation=90, padding=5)

                y_min, y_max = ax_list[j].get_ylim()
                ax_list[j].set_ylim((y_min, y_max+ratio_more*(y_max-y_min)))
                ax_list[j].set_xticks([0, 1, 2, 3])
                ax_list[j].set_xticklabels([
                    r"\algoours", r"\algocatoni",
                    r"\algorivasplata", r"\algoblanchard"],
                    fontsize=9, rotation=45)

            elif(comp_type == "slides"):
                ax_list[j].set_xticks([0, 2])
                ax_list[j].set_xticklabels([
                    r"Ours", r"Rivasplata"],
                    fontsize=12)

        handles, labels = ax_list[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   frameon=False, ncol=len(labels))

        for ax, data in zip(ax_list[:], data_list):
            ax.annotate(data, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')

        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        fig.tight_layout(rect=[0, 0.0, 1, 1])

        os.makedirs("figures/", exist_ok=True)
        plt.savefig(f"figures/comparison_dis_bound_{comp_type}.pdf",
                    bbox_inches="tight")
        del data_
