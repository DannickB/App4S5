import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane

#T(x, y) = (2x, 0.5y)
M = [0.5, 2]
plt.gray()
img = mpimg.imread('../img/goldhill.png')
plt.imshow(img)
plt.show()
new_img = np.zeros([int(img.shape[0]*M[0]), int(img.shape[1]*M[1])])
for x in range(img.shape[0]):
    for y in range(img.shape[0]):
        bruh = img[x][y]
        new_img[int(x*M[0])][int(y*M[1])] = img[x][y]
plt.imshow(new_img)
plt.show()
pass
