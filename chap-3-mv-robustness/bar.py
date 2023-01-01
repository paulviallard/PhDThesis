#!/usr/bin/env python
# Author: Paul VIALLARD
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

###############################################################################

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latex/")
path = os.path.abspath(path)+"/"

f = open("../latex/header_standalone.tex", "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)

plt.rcParams.update({
    "font.size": 12,
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

attack_list = [
    "nothing", "uniformnoise", "uniformnoisepgd",
    "uniformnoiseiterativefgsm", "pgd", "iterativefgsm"]


def convert_name_into_pair(name):
    name_list = name.split("_")
    pair_list = []
    for i in range(len(name_list)):
        if(name_list[i] in attack_list):
            pair_list.append(name_list[i])
    return pair_list


def convert_name_into_dataset(name):
    name_list = name.split("_")
    name = "_".join(name_list[0:3])
    return name


bnd = []
tv = []

csv_list = glob.glob("result_bar/*.csv")
for i in range(len(csv_list)):
    csv_list[i] = pd.read_csv(csv_list[i], index_col=0)
    csv_list[i] = csv_list[i].to_dict()
    bnd.append(csv_list[i]["boundth7"])
    tv.append(csv_list[i]["tv_boundth7"])

for i in range(len(bnd)):
    name_list = bnd[i].keys()

    new_bnd = {}
    new_tv = {}

    for name in name_list:
        new_name = tuple(convert_name_into_pair(name))

        if(new_name[1] != "nothing"):
            new_bnd[new_name] = bnd[i][name]
            new_tv[new_name] = tv[i][name]

    bnd[i] = new_bnd
    tv[i] = new_tv

###############################################################################

sns.set(style="white", rc={"lines.linewidth": 3})
#  fig, ax_list = plt.subplots(
#      1, 2, figsize=(20.0, 3.65), subplot_kw={'xticks': []})
fig, ax_list = plt.subplots(
    2, 1, figsize=(10.0, 2*3.65), subplot_kw={'xticks': []})
latex_dict = {
    'uniformnoise': r"\U",
    'uniformnoisepgd': r"\PGDU",
    'uniformnoiseiterativefgsm': r"\IFGSMU",
    'nothing': '---'
}

bar_name_list = []
for name in bnd[0].keys():
    bar_name_list.append("("+latex_dict[name[0]]+", "+latex_dict[name[1]]+")")


# https://moonbooks.org/Articles/How-to-add-text-on-a-bar-with-matplotlib-/
def autolabel(rects):
    for idx, rect in enumerate(bar_plot):
        height = rect.get_height()
        if(height > 0.45):
            ax.text(rect.get_x() + rect.get_width()/2., 0.05*height,
                    bar_name_list[idx],
                    ha='center', va='bottom', rotation=90,
                    fontsize=21, color="white")
        else:
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    bar_name_list[idx],
                    ha='center', va='bottom', rotation=90, fontsize=21)


ax = ax_list[0]
ax.set_xlabel(r"Hypothesis set $\Hsigned$", fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=21)
bar_plot = ax.bar(bar_name_list,
                  list(bnd[0].values()), color=BLUE)
autolabel(bar_plot)
ax.bar(bar_name_list,
       list(tv[0].values()), color=ORANGE)
ax = ax_list[1]
ax.set_xlabel(r"Hypothesis set $\H$", fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=21)
bar_plot = ax.bar(bar_name_list,
                  list(bnd[1].values()), color=BLUE)
autolabel(bar_plot)
ax.bar(bar_name_list,
       list(tv[1].values()), color=ORANGE)
plt.tight_layout()
fig.savefig("figures/bar.pdf", bbox_inches="tight")
