from PIL import Image
import numpy as np

im = Image.open("figures/cat_original.jpg")
# (left, top, right, bottom)
im = im.crop((300, 1400, 3100, 3600))
im.save("figures/cat.jpg")

rand = np.random.RandomState(0)
rand_im = rand.rand(im.height, im.width, 3)
rand_im = (rand_im*255).astype(np.uint8)
print(rand_im)
rand_im = Image.fromarray(rand_im)
rand_im.save("figures/random.jpg")
