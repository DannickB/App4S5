# -*- coding: utf-8 -*-
"""
S5 GI APP4
Examen formatif pratique

Eric Plourde
Département de génie électrique et de génie informatique
Université de Sherbrooke

ver 01: 20 novembre 2019
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from zplane import zplane

# QUESTION 1

Fe = 8000
Wp = [1000/(Fe/2), 2000/(Fe/2)]
Ws = [750/(Fe/2), 2250/(Fe/2)]
Rp = 0.5
Rs = 40

# évaluer l'ordre du filtre qui respecte les contraintes
b_ord, Wn = signal.cheb2ord(Wp,Ws,Rp,Rs)
# print(b_ord)

# évaluer les paramètres du filtre
num, den = signal.cheby2(b_ord, Rs, Wn, 'bandpass')

# évaluer et afficher la réponse en fréquence du filtre
w, h = signal.freqz(num,den)
fig = plt.figure()
plt.title('QUESTION 1: Réponse en fréquence')
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Fréquence [rad/éch.]')
plt.axis('tight')
#plt.show()
plt.savefig('ReponseFreq.png')

# Vous auriez pu également concevoir un filtre passe-bas, suivi
# d'un filtre passe-haut et multiplier les deux fonctions de transfert
# pour obtenir la fonction de transfert du passe-bande.


# QUESTION 2

# parametres du filtre
b = [1, 0, 0, 0, 0.7];
a = [1, 0, 0, 0, 0.5];

# évaluation et affichage des poles et des zéros
zeros = np.roots(b)
poles = np.roots(a)
zplane(b, a)
print(zeros)
print(poles)

# afficher la réponse en fréquence
w, h = signal.freqz(b,a)
fig = plt.figure()
plt.title('QUESTION 2: Réponse en fréquence')
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Fréquence [rad/éch.]')
plt.axis('tight')
plt.show()

# création de l'impulsion
long_imp = 200
impul = np.zeros(long_imp)
impul[int(long_imp/4)] = 1

# filtrage de l'implusion par le filtre
y = signal.lfilter(b, a, impul)
fig = plt.figure()
plt.title('QUESTION 2: Réponse impulsionnelle')
plt.stem(y[int(long_imp/4):int(long_imp/4)+99])
# plt.plot(y)
plt.xlabel('éch.')
plt.show()
