import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane

fe = 48000
f=2500
f2 = 3500
gain = 0.01
N, wn = signal.buttord(wp=f, ws=f2, gpass=0.2, gstop=40, fs=fe)
print("Filter order: " + str(N))
b, a = signal.butter(N, wn, fs=fe)
w, h = signal.freqz(b, a)
plt.figure("Hw")
plt.title("H(w)")
hdb = 20*np.log10(abs(h))
plt.plot(w, hdb)
plt.figure("Phase")
plt.title("Phase")
plt.plot(w, np.unwrap(np.angle(h)))
plt.figure("Zero/pole")
plt.title("Zero/pole")
z, p, k = zplane(b, a)
plt.show()