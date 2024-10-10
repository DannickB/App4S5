import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane
aberration_path = "./img/goldhill_aberrations.npy"

#Image reading

img_aberration = np.load(aberration_path)
plt.figure("Aberration")
plt.gray()
plt.title("Image avec aberration")
plt.imshow(img_aberration)

#fonction de transfere inverse
z1 = 0
z2 = -0.99
z3 = -0.99
z4 = 0.8

p1 = 0.9*np.exp(1j*np.pi/2)
p2 = 0.9*np.exp(-1j*np.pi/2)
p3 = 0.95*np.exp(1j*np.pi/8)
p4 = 0.95*np.exp(-1j*np.pi/8)

b = np.poly([z1, z2, z3, z4])
a = np.poly([p1, p2, p3, p4])

#zero et poles
plt.figure("Pz map H(z)")
plt.title("Pz map H(z)")
zplane(a, b)
plt.figure("Pz map H(z) inverse")
plt.title("Pz map H(z) inverse")
z, p, k = zplane(b, a)

#filtering image
img_clean = signal.lfilter(b, a, img_aberration)
plt.figure("Image cleaned")
plt.title("Image cleaned")
plt.imshow(img_clean)

plt.show()
