from scipy import signal
import numpy as np
import matplotlib. pyplot as plt
from zplane import zplane

fs = 8000
gpass = 0.5
gstop = 40
wp = [1000, 2000]
ws = [750, 2250]
N, wn = signal.cheb2ord(wp, ws, gpass, gstop, fs=fs)
b, a = signal.cheby2(N, Wn=wn, rs=gstop, fs=fs, output='ba', btype='bandpass')
w, h = signal.freqz(b, a, fs=fs)
plt.figure("Q1")
plt.title("Q1")
plt.plot(w, 20*np.log10(abs(h)))
#Q2
z = np.roots([1, 0, 0, 0, 0.7])
p = np.roots([1, 0, 0, 0, 0.5])

b = np.poly(z)
a = np.poly(p)
plt.figure("Q2 a")
plt.title("Q2 a")
zplane(b, a, "zp map")

w, h = signal.freqz(b, a)
plt.figure("Q2 b")
plt.title("Q2 b")
plt.plot(w, 20*np.log10(abs(h)))

d = np.zeros(100)
d[0] = 1

y = signal.lfilter(b, a, d)
plt.figure("Q2 c")
plt.title("Q2 c")
plt.stem(y)



plt.show()