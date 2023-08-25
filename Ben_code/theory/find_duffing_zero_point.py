# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:21:36 2020

@author: Ben
"""


import numpy as np
import band_fitting_routines as bandfitr
import plotting_routines as plotr
import matplotlib.pyplot as plt


n_charges = 3
f0_bare = 5.757E9
fc = 54.8E9
fj = 14.9E9
phizp = 0.0878776
n_points = 30
ngs = np.linspace(-0.6,0.6,n_points)
phi_edge = 0.25*np.pi
phis = np.linspace(-phi_edge,phi_edge,n_points)
d2f = bandfitr.dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
f0_theory = f0_bare + ((phizp**2)*d2f)

fig = plotr.plot_band_colormap(phis,ngs,f0_theory,title='f0 Theory')
plt.show()


duffing_theory = ((phizp**4)/2)*bandfitr.dfCPT(ngs,phis,fc,fj,4,n_charges=n_charges)

fig = plotr.plot_band_colormap(phis,ngs,duffing_theory,title='duffing nonlinearity')
plt.show()


plt.plot(f0_theory.ravel(),duffing_theory.ravel(),'.')
plt.xlabel('f0')
plt.ylabel('duffing')
plt.title('duffing nonlinearity vs resonant frequency')
plt.show()

plt.plot(f0_theory.ravel(),duffing_theory.ravel()**2,'.')
plt.xlabel('f0')
plt.ylabel('duffing squared')
plt.title('duffing nonlinearity squared vs resonant frequency')
plt.show()

min_duffing = np.amin(np.abs(duffing_theory))
min_duffing_args = np.argmin(np.abs(duffing_theory))
print('min abs(duffing) = '+str(min_duffing))
print('min abs(duffing) args = '+str(min_duffing_args))

print('phi = -'+str(phi_edge/np.pi)+' pi')
print('f0:')
plt.plot(ngs,f0_theory[0],'.')
plt.xlabel('ng')
plt.ylabel('f0')
plt.show()
print('duffing:')
plt.plot(ngs,duffing_theory[0],'.')
plt.xlabel('ng')
plt.ylabel('duffing')
plt.show()

print('phi = +'+str(phi_edge/np.pi)+' pi')
print('f0:')
plt.plot(ngs,f0_theory[-1],'.')
plt.xlabel('ng')
plt.ylabel('f0')
plt.show()
print('duffing:')
plt.plot(ngs,duffing_theory[-1],'.')
plt.xlabel('ng')
plt.ylabel('duffing')
plt.show()