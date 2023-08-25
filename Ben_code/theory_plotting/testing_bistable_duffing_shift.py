# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 22:55:40 2019

@author: Ben
"""


import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def duffing_discriminant(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]
    discriminant = 18*a*b*c*d - \
                    4*(b**3)*d + \
                    (b**2)*(c**2) - \
                    4*a*(c**3) - \
                    27*(a**2)*(d**2)
    return discriminant

def gamma_duffing(df, chi, kerr, Nin, imag_cutoff=1E-30):
    response = []
    discriminants = []
    bistability_flag = 0
    prev_root = None
    for freq in df:
        coeffs = np.array([kerr**2, \
                           -2*kerr*freq, \
                           (freq**2)+(1/4), \
                           -1*(chi/(1+chi))*Nin])
        discriminants.append(duffing_discriminant(coeffs))
        roots = np.roots(coeffs)
#        print(roots)
        im_part = np.imag(roots)
        real_roots = np.real(roots[np.where(np.abs(im_part)<imag_cutoff)])
#        print('im part: ')
#        print(im_part)
#        print('real roots:')
#        print(real_roots)
#        input()
        if len(real_roots) > 1:
            if bistability_flag == 0:
                print('MORE THAN ONE REAL ROOT!')
                bistability_flag = 1
            if prev_root:
                root = real_roots[np.argmin(np.abs(real_roots-prev_root))]
            else:
                root = real_roots[np.argmin(real_roots)]
        else:
            root = real_roots[0]
        response.append(root)
        prev_root = root
    response = np.array(response)
    discriminants = np.array(discriminants)
    numerator = (df-(kerr*response))-(1j*(((1-chi)/(1+chi))/2))
    denominator = (df-(kerr*response))-(1j/2)
    gamma = numerator/denominator
    return gamma, discriminants


scale = 12
ticklabelsize = 14
labelsize = 16
plotlabelsize = 20

freq_lim = 30
freq_stride = 100
df = np.linspace(-1*freq_lim,freq_lim,5001)
chi = 0.9
Nins = np.linspace(0.1,200,20)
kerr = -10

min_mag_freq_list = []
for ii in range(len(Nins)):
    Nin = Nins[ii]
    gamma = gamma_duffing(df,chi,kerr,Nin)[0]
    x = np.real(gamma)
    x[np.where(x==0)]=+0.0
    y = np.imag(gamma)
    mag = np.sqrt((x**2)+(y**2))
    arg = np.arctan2(y,x)
    min_mag_freq_list.append(df[np.argmin(mag)])


    
plt.plot(Nins,min_mag_freq_list,linestyle='None',marker='.')
plt.show()
