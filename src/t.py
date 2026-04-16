from PIL import Image
import numpy as np

im = Image.open("../images/monkey_hero.png")
arr = np.asarray(im)

print(arr.shape)
