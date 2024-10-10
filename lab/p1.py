import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane

#a)
K = 1
z1 = 0.8j
z2 = -0.8j
p1 = 0.95*np.exp(1j*np.pi/8)
p2 = 0.95*np.exp(-1j*np.pi/8)
b = np.poly([z1, z2])
a = np.poly([p1, p2])
z, p, k = zplane(b, a)
#b) - oui
#c)
w, h = signal.freqz(b, a)
plt.figure("Hw")
plt.title("H(w)")
hdb = 20*np.log10(h)
plt.plot(w, hdb)

plt.savefig("P1_c.png")
#d)
z = np.arange(1000)
Hz = (z-z1)*(z-z2)/((z-p1)*(z-p2))
d = np.zeros(len(z))
d[int(len(z)/2)] = 1
hn = signal.lfilter(b, a, d)
hm = np.fft.fft(hn)
plt.figure("h[m]")
plt.plot(20*np.log10(hm[0:int(len(z)/2)]))
#e) - H(w)^-1
hn2 = signal.lfilter(a, b, hn)
plt.figure("2eme filtre")
plt.plot(hn2)
plt.show()
pass