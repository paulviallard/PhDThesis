import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
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


def latex_fun(x, r, c, data):
    return x

###############################################################################


data_ = NDData("sourcecode/exp.csv")

for prior in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    lr_dict = {
        0.000001: r"$\lr=10^{-6}$",
        0.0001: r"$\lr=10^{-4}$"}
    data_dict = {
        "mnist": r"\rotatebox[origin=c]{90}{\small{MNIST}}",
        "fashion": r"\rotatebox[origin=c]{90}{\small{Fashion}}",
        "cifar10": r"\rotatebox[origin=c]{90}{\small{CIFAR-10}}"}
    var_dict = {
        0.000001: r"\rotatebox[origin=c]{90}{\small{$\sigma^2=10^{-6}$}}",
        0.00001: r"\rotatebox[origin=c]{90}{\small{$\sigma^2=10^{-5}$}}",
        0.0001: r"\rotatebox[origin=c]{90}{\small{$\sigma^2=10^{-4}$}}",
        0.001: r"\rotatebox[origin=c]{90}{\small{$\sigma^2=10^{-3}$}}"}
    bound_dict = {
        "ours": r"\algoours",
        "rivasplata": r"\algorivasplata",
        "blanchard": r"\algoblanchard",
        "catoni": r"\algocatoni",
        "stoNN": r"\algostoNN",
    }

    for data in data_dict.keys():

        table_str = ""
        i = 0
        j = 0

        for var in var_dict.keys():

            d_prior = data_.get(
                "data", "prior",
                "test-risk-mean", "test-risk-std",
                "post-risk-mean", "post-risk-std",
                "bound-ours-mean", "bound-ours-std",
                "bound-blanchard-mean", "bound-blanchard-std",
                "bound-catoni-mean", "bound-catoni-std",
                "bound-rivasplata-mean", "bound-rivasplata-std",
                var=var, step="prior", data=data)

            d_prior[r"$\Risk_{\Tcal}(h)$"] = [
                r"{:.3f} $\pm$ {:.3f}".format(
                    d_prior["test-risk-mean"][i],
                    d_prior["test-risk-std"][i]).replace("0.", ".")
                for i in range(len(d_prior))]
            d_prior[r"$\Risk_{\Scal}(h)$"] = [
                r"{:.3f} $\pm$ {:.3f}".format(
                    d_prior["post-risk-mean"][i],
                    d_prior["post-risk-std"][i]).replace("0.", ".")
                for i in range(len(d_prior))]
            d_prior[r"\cref{chap:dis-pra:corollary:nn}"] = [
                r"{:.3f} $\pm$ {:.3f}".format(
                    d_prior["bound-ours-mean"][i],
                    d_prior["bound-ours-std"][i]).replace("0.", ".")
                for i in range(len(d_prior))]
            d_prior[r"\cref{chap:dis-pra:eq:nn-rivasplata}"] = [
                r"{:.3f} $\pm$ {:.3f}".format(
                    d_prior["bound-rivasplata-mean"][i],
                    d_prior["bound-rivasplata-std"][i]).replace("0.", ".")
                for i in range(len(d_prior))]
            d_prior[r"\cref{chap:dis-pra:eq:nn-blanchard}"] = [
                r"{:.3f} $\pm$ {:.3f}".format(
                    d_prior["bound-blanchard-mean"][i],
                    d_prior["bound-blanchard-std"][i]).replace("0.", ".")
                for i in range(len(d_prior))]
            d_prior[r"\cref{chap:dis-pra:eq:nn-catoni}"] = [
                r"{:.3f} $\pm$ {:.3f}".format(
                    d_prior["bound-catoni-mean"][i],
                    d_prior["bound-catoni-std"][i]).replace("0.", ".")
                for i in range(len(d_prior))]

            del d_prior["data"]
            del d_prior["test-risk-mean"]
            del d_prior["test-risk-std"]
            del d_prior["post-risk-mean"]
            del d_prior["post-risk-std"]
            del d_prior["bound-ours-mean"]
            del d_prior["bound-ours-std"]
            del d_prior["bound-blanchard-mean"]
            del d_prior["bound-blanchard-std"]
            del d_prior["bound-catoni-mean"]
            del d_prior["bound-catoni-std"]
            del d_prior["bound-rivasplata-mean"]
            del d_prior["bound-rivasplata-std"]

            d_prior.index = [(var_dict[var],
                              str(d_prior["prior"][i]).replace("0.", "."))
                             for i in range(len(d_prior))]
            d_prior = d_prior.sort_values(by=["prior"])

            del d_prior["prior"]
            d_prior = d_prior.rename_axis(
                ["Split"], axis=1)
            d_prior.index = pd.MultiIndex.from_tuples(d_prior.index)

            tmp_str = NDData.to_latex(
                d_prior, latex_fun,
                col_format="cccccccc")

            if(table_str == ""):
                table_str = "\n".join(
                    tmp_str.split("\n")[:4])+"\n"
            table_str += "\n".join(
                tmp_str.split("\n")[4:-3])+"\n\\midrule\n"

            i += 1
            if(i == 2):
                i = 0
                j += 1
                table_str = "\n".join(table_str.split("\n")[:-2])
                table_str += "\n\\bottomrule\n\\end{tabular}"

                os.makedirs("tables/", exist_ok=True)
                with open(f"tables/table_2_data_{data}_{j}.tex", "w") as f:
                    f.write(table_str)
                table_str = ""

###############################################################################
