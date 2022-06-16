import numpy as np
import matplotlib.pyplot as plt

from lib import tally

dimension = (200, 200, 1)
image_3d = tally.usrbin("test_22.bnn", dimension)
image_2d = image_3d[:,:,0]

plt.imshow(image_2d, cmap="gray")
plt.colorbar()
plt.xlabel("y pixel index")
plt.ylabel("x pixel index")
plt.show()
