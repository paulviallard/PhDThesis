import os
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap

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

scatter_cmap = LinearSegmentedColormap.from_list(
    "scatter_cmap", [BLUE, RED])
scatter_bg_cmap = LinearSegmentedColormap.from_list(
    "scatter_bg_cmap", [BLUE, WHITE])

###############################################################################


def generalize(x_):
    return (-(1.9/0.6)*x_[:, 0]+1.9 >= x_[:, 1]).astype(float)

# --------------------------------------------------------------------------- #


input_label = torchvision.datasets.MNIST(
    root=".", train=False, download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
)

label = np.array(input_label.targets)
input = np.array(input_label.data)
input_0 = input[label == 0][:3]
input_5 = input[label == 5][1:4]
input = np.concatenate((input_5, input_0), axis=0)

x_0 = np.array([[0.0, 0.0],
                [0.1, 1.0],
                [0.2, 0.5]])
x_5 = np.array([[0.0+0.5, 0.0+1.0],
                [0.3+0.5, 0.2+1.0],
                [0.2+0.5, 0.5+1.0]])
x = np.concatenate((x_0, x_5), axis=0)

# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 1, figsize=(3, 3))

for i in range(len(input)):
    image = input[i].reshape((1, 28, 28))
    image = image.repeat(4, axis=0)
    image[3, :, :] = 255*(image[3, :, :] > 0).astype(int)
    image[:3, :, :] = 255-image[:3, :, :]
    image = image.swapaxes(0, 2)
    image = image.swapaxes(0, 1)

    image = OffsetImage(image, zoom=0.7, cmap="gray", alpha=1.0)
    a = AnnotationBbox(
        image, (x[i, 0], x[i, 1]), frameon=False, zorder=10)
    ax.add_artist(a)

ax.set_xlim(-0.05, 0.85)
ax.set_ylim(-0.4, 1.9)
ax.tick_params(left=False, labelleft=False,
               labelbottom=False, bottom=False)

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/introduction_1.pdf", bbox_inches="tight")

x_1, x_2 = np.meshgrid(
    np.arange(-0.05, 0.85, 0.0005),
    np.arange(-0.4, 1.9, 0.0005))
x_ = np.concatenate((x_1.reshape((-1, 1)), x_2.reshape((-1, 1))), axis=1)
y_ = generalize(x_).reshape(x_1.shape)
ax.contourf(x_1, x_2, y_, cmap=scatter_bg_cmap, alpha=0.8)
ax.plot([0.0, 1.0], [-(1.9/0.6)*0.0+1.9, -(1.9/0.6)*1.0+1.9],
        "-", linewidth=4, c=BLACK)

fig.savefig("figures/introduction_2.pdf", bbox_inches="tight")
