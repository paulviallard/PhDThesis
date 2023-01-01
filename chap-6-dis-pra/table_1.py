import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import warnings
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
        0.000001: r"$\sigma^2=10^{-6}$",
        0.00001: r"$\sigma^2=10^{-5}$",
        0.0001: r"$\sigma^2=10^{-4}$",
        0.001: r"$\sigma^2=10^{-3}$"}
    bound_dict = {
        "ours": r"\algoours",
        "rivasplata": r"\algorivasplata",
        "blanchard": r"\algoblanchard",
        "catoni": r"\algocatoni",
        "stoNN": r"\algostoNN",
    }

    table_str = ""
    for lr in lr_dict.keys():

        new_lr = True

        for data in data_dict.keys():

            data_all = None

            for var in var_dict.keys():

                var_str = var_dict[var]
                d_ours = data_.get(
                    "data", "bound", "bound-ours-mean", "bound-ours-std",
                    "test-risk-mean", "test-risk-std",
                    "post-risk-mean", "post-risk-std", "div-renyi",
                    "bound-kl-sample", "div-kl",
                    var=var, prior=prior, post_lr=lr, data=data, bound="ours")

                d_sto = pd.DataFrame(data={
                    (var_str, r"$\Risk_{\Tcal}(h)$"): [r"\textemdash"],
                    (var_str, "Bnd"): [
                        "{:.3f}".format(d_ours["bound-kl-sample"][0]
                                        ).replace("0.", ".")],
                    (var_str, r"$\Risk_{\Scal}(h)$"): [r"\textemdash"],
                    (var_str, "Div"): ["{:.3f}".format(d_ours["div-kl"][0]
                                                       ).replace("0.", ".")],
                }, index=[(data_dict[data], bound_dict["stoNN"])])

                d_ours[(var_str, r"$\Risk_{\Tcal}(h)$")] = [
                    r"{:.3f} $\pm$ {:.3f}".format(
                        d_ours["test-risk-mean"][0],
                        d_ours["test-risk-std"][0]).replace("0.", ".")]
                d_ours[(var_str, "Bnd")] = [r"{:.3f} $\pm$ {:.3f}".format(
                    d_ours["bound-ours-mean"][0],
                    d_ours["bound-ours-std"][0]).replace("0.", ".")]
                d_ours[(var_str, r"$\Risk_{\Scal}(h)$")] = [
                    r"{:.3f} $\pm$ {:.3f}".format(
                        d_ours["post-risk-mean"][0],
                        d_ours["post-risk-std"][0]).replace("0.", ".")]

                d_ours[(var_str, "Div")] = [
                    "{:.3f}".format(d_ours["div-renyi"][0]).replace("0.", ".")]
                d_ours.index = [(data_dict[data], bound_dict["ours"])]
                del d_ours["bound-ours-mean"]
                del d_ours["bound-ours-std"]
                del d_ours["post-risk-mean"]
                del d_ours["post-risk-std"]
                del d_ours["test-risk-mean"]
                del d_ours["test-risk-std"]
                del d_ours["div-renyi"]
                del d_ours["bound-kl-sample"]
                del d_ours["div-kl"]

                d_bla = data_.get(
                    "data", "bound", "bound-blanchard-mean",
                    "bound-blanchard-std",
                    "post-risk-mean", "post-risk-std",
                    "test-risk-mean", "test-risk-std",
                    "div-rivasplata-mean", "div-rivasplata-std",
                    var=var, prior=prior, post_lr=lr, data=data,
                    bound="blanchard")
                d_bla[(var_str, r"$\Risk_{\Tcal}(h)$")] = [
                    r"{:.3f} $\pm$ {:.3f}".format(
                        d_bla["test-risk-mean"][0],
                        d_bla["test-risk-std"][0]).replace("0.", ".")]
                d_bla[(var_str, "Bnd")] = [r"{:.3f} $\pm$ {:.3f}".format(
                    d_bla["bound-blanchard-mean"][0],
                    d_bla["bound-blanchard-std"][0]).replace("0.", ".")]
                d_bla[(var_str, r"$\Risk_{\Scal}(h)$")] = [
                    r"{:.3f} $\pm$ {:.3f}".format(
                        d_bla["post-risk-mean"][0],
                        d_bla["post-risk-std"][0]).replace("0.", ".")]
                d_bla[(var_str, "Div")] = [r"{:.3f} $\pm$ {:.3f}".format(
                    d_bla["div-rivasplata-mean"][0],
                    d_bla["div-rivasplata-std"][0]).replace("0.", ".")]
                d_bla.index = [(data_dict[data], bound_dict["blanchard"])]
                del d_bla["bound-blanchard-mean"]
                del d_bla["bound-blanchard-std"]
                del d_bla["post-risk-mean"]
                del d_bla["post-risk-std"]
                del d_bla["test-risk-mean"]
                del d_bla["test-risk-std"]
                del d_bla["div-rivasplata-mean"]
                del d_bla["div-rivasplata-std"]

                d_cat = data_.get(
                    "data", "bound", "bound-catoni-mean", "bound-catoni-std",
                    "post-risk-mean", "post-risk-std",
                    "test-risk-mean", "test-risk-std",
                    "div-rivasplata-mean", "div-rivasplata-std",
                    var=var, prior=prior, post_lr=lr, data=data,
                    bound="catoni")
                d_cat[(var_str, r"$\Risk_{\Tcal}(h)$")] = [
                    r"{:.3f} $\pm$ {:.3f}".format(
                        d_cat["test-risk-mean"][0],
                        d_cat["test-risk-std"][0]).replace("0.", ".")]
                d_cat[(var_str, "Bnd")] = [r"{:.3f} $\pm$ {:.3f}".format(
                    d_cat["bound-catoni-mean"][0],
                    d_cat["bound-catoni-std"][0]).replace("0.", ".")]
                d_cat[(var_str, r"$\Risk_{\Scal}(h)$")] = [
                    r"{:.3f} $\pm$ {:.3f}".format(
                        d_cat["post-risk-mean"][0],
                        d_cat["post-risk-std"][0]).replace("0.", ".")]
                d_cat[(var_str, "Div")] = [r"{:.3f} $\pm$ {:.3f}".format(
                    d_cat["div-rivasplata-mean"][0],
                    d_cat["div-rivasplata-std"][0]).replace("0.", ".")]
                d_cat.index = [(data_dict[data], bound_dict["catoni"])]
                del d_cat["bound-catoni-mean"]
                del d_cat["bound-catoni-std"]
                del d_cat["post-risk-mean"]
                del d_cat["post-risk-std"]
                del d_cat["test-risk-mean"]
                del d_cat["test-risk-std"]
                del d_cat["div-rivasplata-mean"]
                del d_cat["div-rivasplata-std"]

                d_riv = data_.get(
                    "data", "bound", "bound-rivasplata-mean",
                    "bound-rivasplata-std",
                    "post-risk-mean", "post-risk-std",
                    "test-risk-mean", "test-risk-std",
                    "div-rivasplata-mean", "div-rivasplata-std",
                    var=var, prior=prior, post_lr=lr, data=data,
                    bound="rivasplata")
                d_riv[(var_str, r"$\Risk_{\Tcal}(h)$")] = [
                    r"{:.3f} $\pm$ {:.3f}".format(
                        d_riv["test-risk-mean"][0],
                        d_riv["test-risk-std"][0]).replace("0.", ".")]
                d_riv[(var_str, "Bnd")] = [r"{:.3f} $\pm$ {:.3f}".format(
                    d_riv["bound-rivasplata-mean"][0],
                    d_riv["bound-rivasplata-std"][0]).replace("0.", ".")]
                d_riv[(var_str, r"$\Risk_{\Scal}(h)$")] = [
                    r"{:.3f} $\pm$ {:.3f}".format(
                        d_riv["post-risk-mean"][0],
                        d_riv["post-risk-std"][0]).replace("0.", ".")]
                d_riv[(var_str, "Div")] = [r"{:.3f} $\pm$ {:.3f}".format(
                    d_riv["div-rivasplata-mean"][0],
                    d_riv["div-rivasplata-std"][0]).replace("0.", ".")]
                d_riv.index = [(data_dict[data], bound_dict["rivasplata"])]
                del d_riv["bound-rivasplata-mean"]
                del d_riv["bound-rivasplata-std"]
                del d_riv["post-risk-mean"]
                del d_riv["post-risk-std"]
                del d_riv["test-risk-mean"]
                del d_riv["test-risk-std"]
                del d_riv["div-rivasplata-mean"]
                del d_riv["div-rivasplata-std"]

                tmp_data = pd.concat([d_ours, d_bla, d_cat, d_riv, d_sto])
                tmp_data.rename(columns={
                    "data": ("test", "data"),
                    "bound": ("test", "bound")}, inplace=True)
                tmp_data.columns = pd.MultiIndex.from_tuples(tmp_data.columns)
                tmp_data.index = pd.MultiIndex.from_tuples(tmp_data.index)

                if(data_all is None):
                    data_all = tmp_data
                else:
                    data_all = pd.concat(
                        [data_all, tmp_data], axis=1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                del data_all[("test", "data")]
                del data_all[("test", "bound")]

            data_all = data_all.rename_axis(
                ["", lr_dict[lr]], axis=1)

            tmp_str = NDData.to_latex(
                data_all, latex_fun,
                col_format="rr|clcl|clcl|clcl|clcl")
            tmp_str = tmp_str.replace(
                r"\multicolumn{4}{r}", r"\multicolumn{4}{c}")

            tmp_str = tmp_str.split("\n")
            tmp_str.insert(3, r"\midrule")
            tmp_str = "\n".join(tmp_str)
            if(table_str == ""):
                table_str = "\n".join(
                    tmp_str.split("\n")[:2])+"\n"
            if(new_lr):
                new_lr = False
                table_str += "\n".join(
                    tmp_str.split("\n")[2:-3])+"\n\\midrule\n"
            else:
                table_str += "\n".join(
                    tmp_str.split("\n")[6:11])+"\n\\midrule\n"

    table_str = "\n".join(table_str.split("\n")[:-2])
    table_str += "\n\\bottomrule\n\\end{tabular}"

    os.makedirs("tables/", exist_ok=True)
    with open(f"tables/table_1_prior_{prior}.tex", "w") as f:
        f.write(table_str)

    ###########################################################################
