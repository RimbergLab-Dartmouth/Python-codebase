# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:51:23 2020

@author: Ben
"""

import numpy as np
import band_fitting_routines as bandfitr
import plotting_routines as plotr
import matplotlib.pyplot as plt


n_charges = 2
f0_bare = 5.757E9
fc = 54.8E9
fj = 14.9E9
phizp = 0.0878776
ng0 = 0.65
phi0 = np.pi/2.0
n_points = 13
ngs = np.linspace(-ng0,ng0,n_points)
phis = np.linspace(-phi0,phi0,n_points)
d2f = bandfitr.dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
f0_theory = f0_bare + ((phizp**2)*d2f)

fig = plotr.plot_band_colormap(phis,ngs,f0_theory,title='f0 Theory')
plt.show()


nonlinear6 = (phizp**6)*bandfitr.dfCPT(ngs,phis,fc,fj,6,n_charges=n_charges)

fig = plotr.plot_band_colormap(phis,ngs,nonlinear6,title='6th order nonlinearity')
plt.show()


plt.plot(f0_theory.ravel(),nonlinear6.ravel(),'.')
plt.xlabel('f0')
plt.ylabel('duffing')
plt.title('6th order nonlinearity vs resonant frequency')
plt.show()

plt.plot(f0_theory.ravel(),nonlinear6.ravel()**2,'.')
plt.xlabel('f0')
plt.ylabel('duffing squared')
plt.title('6th order nonlinearity squared vs resonant frequency')
plt.show()

