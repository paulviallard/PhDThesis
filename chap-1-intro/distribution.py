import os
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity as density
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

input_label_train = torchvision.datasets.MNIST(
    root=".", train=True, download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
)

input_train = np.array(input_label_train.data)
label_train = np.array(input_label_train.targets)

select = (
    (label_train == 5)+(label_train == 1)
    + (label_train == 7)+(label_train == 3)+(label_train == 6))
input_train = input_train[select]
label_train = label_train[select]
input_train = input_train[:1000, :]
label_train = label_train[:1000]
input_train = input_train.reshape((1000, 784))
tsne = TSNE(n_components=2, random_state=0, init='random', learning_rate=200.0)
input_tsne = tsne.fit_transform(input_train)
input_tsne = (input_tsne-np.min(input_tsne))/(
    np.max(input_tsne)-np.min(input_tsne))

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

for i in range(150):
    image = input_train[i].reshape((1, 28, 28))
    image = image.repeat(4, axis=0)
    image[3, :, :] = 255*(image[3, :, :] > 0).astype(int)
    image[:3, :, :] = 255-image[:3, :, :]
    image = image.swapaxes(0, 2)
    image = image.swapaxes(0, 1)

    image = OffsetImage(image, zoom=0.7, cmap="gray", alpha=1.0)
    a = AnnotationBbox(
        image, (input_tsne[i, 0], input_tsne[i, 1]), frameon=False, zorder=10)
    ax.add_artist(a)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.02, 1.03)
ax.axis("off")

density = density(
    bandwidth=0.04, metric="haversine",
    kernel="gaussian", algorithm="ball_tree").fit(input_tsne)

x_1, x_2 = np.meshgrid(
    np.arange(-0.05, 1.05, 0.005),
    np.arange(-0.05, 1.05, 0.005))

x = np.concatenate((x_1.reshape((-1, 1)), x_2.reshape((-1, 1))), axis=1)
d = density.score_samples(x).reshape(x_1.shape)
ax.contour(x_1, x_2, np.exp(d))

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/distribution.pdf", bbox_inches="tight")

# --------------------------------------------------------------------------- #

pwd = os.path.dirname(__file__)
os.remove(os.path.join(pwd, "MNIST/raw/t10k-labels-idx1-ubyte"))
os.remove(os.path.join(pwd, "MNIST/raw/t10k-labels-idx1-ubyte.gz"))
os.remove(os.path.join(pwd, "MNIST/raw/t10k-images-idx3-ubyte"))
os.remove(os.path.join(pwd, "MNIST/raw/t10k-images-idx3-ubyte.gz"))
os.remove(os.path.join(pwd, "MNIST/raw/train-labels-idx1-ubyte"))
os.remove(os.path.join(pwd, "MNIST/raw/train-labels-idx1-ubyte.gz"))
os.remove(os.path.join(pwd, "MNIST/raw/train-images-idx3-ubyte"))
os.remove(os.path.join(pwd, "MNIST/raw/train-images-idx3-ubyte.gz"))
os.rmdir(os.path.join(pwd, "MNIST/raw/"))
os.rmdir(os.path.join(pwd, "MNIST/"))
