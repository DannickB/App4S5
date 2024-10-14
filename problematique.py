#######################
# bild2707
# labg0902
#######################
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane
img_path = "./img/image_complete.npy"

#Image reading

img_aberration = np.load(img_path)
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

#filtering image
img_rotation = signal.lfilter(b, a, img_aberration)
plt.figure("Image sans aberration")
plt.title("Image sans aberration")
plt.imshow(img_rotation)

#zero et poles
plt.figure("Pz map H(z)")
plt.title("Pz map H(z)")
zplane(a, b)
plt.figure("Pz map H(z) inverse")
plt.title("Pz map H(z) inverse")
z, p, k = zplane(b, a)

#Rotation
plt.figure("Image sans rotation")
plt.title("Image sans rotation")
plt.imshow(img_rotation)
M = np.array([[0, 1], [-1, 0]]) #Matrice de rotation
img_noise = np.zeros([img_rotation.shape[0], img_rotation.shape[1]])
for x in range(img_rotation.shape[0]):
    for y in range(img_rotation.shape[0]):
        coords = np.array([x, y]) #Matrice de coordonnées
        coords = coords.transpose() #Transposer la matrice dans le bon sens
        new_coords = np.matmul(M,coords) #Nouvelles coordonnées avec la matrice de rotation
        new_coords.transpose()
        img_noise[new_coords[0]][new_coords[1]] = img_rotation[x][y]
plt.figure("Image avec rotation")
plt.title("Image avec rotation")
plt.imshow(img_noise)

#Filtre du bruit (Par python)

fe = 1600
f = 500
f2 = 750

Nb, wn_b = signal.buttord(wp=f, ws=f2, gpass=0.2, gstop=60, fs=fe)
Nc1, wn_c1 = signal.cheb1ord(wp=f, ws=f2, gpass=0.2, gstop=60, fs=fe)
Nc2, wn_c2 = signal.cheb2ord(wp=f, ws=f2, gpass=0.2, gstop=60, fs=fe)
Ne, wn_e = signal.ellipord(wp=f, ws=f2, gpass=0.2, gstop=60, fs=fe)
print("Butterworth filter order: " + str(Nb))
print("Chebyshev type I filter order: " + str(Nc1))
print("Chebyshev type 2 filter order: " + str(Nc2))
print("Elliptique filter order: " + str(Ne))
b, a = signal.ellip(Ne, 0.2, 60, wn_e, fs=fe, btype='low')
w, h = signal.freqz(b, a, fs=fe)
img_encode = signal.lfilter(b, a, img_noise)
plt.figure("Image sans noise")
plt.title("Image sans noise")
plt.imshow(img_encode)
plt.figure("Hz")
plt.title("Réponse en fréquence du filtre créé avec Python")
hdb = 20*np.log10(abs(h))
plt.plot(w, hdb)
# plt.figure("Phase")
# plt.title("Phase")
# plt.plot(w, np.unwrap(np.angle(h)))
plt.figure("Zero/pole")
plt.title("Zero/pole")
z, p, k = zplane(b, a)

# Filtre du bruit (mathematiquement)

num = (np.roots([0.418, 0.837, 0.418]))
den = (np.roots([1, 0.463, 0.210]))
a = np.poly(den)
b = np.poly(num)
f, h = signal.freqz(b, a, fs=fe)
hdb = 20 * np.log10(np.abs(h))

img_encode = signal.lfilter(b, a, img_noise)
plt.figure("Image sand noise (math)")
plt.title("Image sans noise (math)")
plt.imshow(img_encode)
plt.figure("Reponse freq (math)")
plt.plot(f, hdb)
plt.title("Réponse en fréquence du filtre calculé mathématiquement")
plt.xlabel("Frequence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.figure("Zero/Pole (math)")
plt.title("Zero/Pole (math)")
z, p, k = zplane(b, a)

# Compression

cov = np.cov(img_encode)
s, v = np.linalg.eig(cov)
v = [v for _, v in sorted(zip(s, v), reverse=True)]
v = np.array(v)
v = v.transpose()
s = sorted(s, reverse=True)
img_decode = np.matmul(v, img_encode)
#image avec 50% de perte
img_50 = np.zeros_like(img_decode)
img_50[0:int(len(img_decode)/2)] = img_decode[0:int(len(img_decode)/2)]
#image avec 70% de perte
img_70 = np.zeros_like(img_decode)
img_70[0:int(len(img_decode)*0.3)] = img_decode[0:int(len(img_decode)*0.3)]

#Recomposition

v_inv = np.linalg.inv(v)
img_50_final = np.matmul(v_inv, img_50)
img_70_final = np.matmul(v_inv, img_70)

plt.figure("Image final 50%")
plt.title("Image final 50%")
plt.imshow(img_50_final)
plt.figure("Image final 70%")
plt.title("Image final 70%")
plt.imshow(img_70_final)


plt.show()
pass