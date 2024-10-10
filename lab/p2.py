import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane

z1 = np.exp(1j*np.pi/16)
z2 = np.exp(-1j*np.pi/16)
p1 = 0.9*np.exp(1j*np.pi/16)
p2 = 0.9*np.exp(-1j*np.pi/16)

b = np.poly([z1, z2])
a = np.poly([p1, p2])
z, p, k = zplane(b, a)
w, h = signal.freqz(b, a)
plt.figure("Hw")
plt.title("H(w)")
hdb = 20*np.log10(h)
plt.plot(w, hdb)
n = np.arange(500)
x = np.sin(n*np.pi/16)+np.sin(n*np.pi/32)
xn = signal.lfilter(b, a, x)
plt.figure("Signals")
plt.title("Signals")
plt.plot(x)
plt.plot(xn)
plt.show()