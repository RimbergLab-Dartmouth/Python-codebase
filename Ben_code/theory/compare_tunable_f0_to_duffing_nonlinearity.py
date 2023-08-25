# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:34:24 2019

@author: Ben
"""

import numpy as np
import band_fitting_routines as bandfitr
import plotting_routines as plotr
import matplotlib.pyplot as plt


n_charges = 2
f0_bare = 5.757E9
fc = 19.0E9
fj = 10.4E9
phizp = 0.0878776
ngs = np.linspace(-0.6,0.6,50)
phis = np.linspace(-np.pi/2,np.pi/2,50)
d2f = bandfitr.dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
f0_theory = f0_bare + ((phizp**2)*d2f)

fig = plotr.plot_band_colormap(phis,ngs,f0_theory,title='f0 Theory')
plt.show()


duffing_theory = ((phizp**4)/2)*bandfitr.dfCPT(ngs,phis,fc,fj,4,n_charges=n_charges)

fig = plotr.plot_band_colormap(phis,ngs,duffing_theory,title='duffing nonlinearity')
plt.show()

fig = plotr.plot_band_colormap(phis,ngs,np.abs(duffing_theory),title='abs(duffing nonlinearity)')
plt.show()

plt.figure(figsize=(10,8))
plt.plot(f0_theory.ravel(),duffing_theory.ravel()/1.0E6,'.')
plt.xlabel('f0')
plt.ylabel('duffing (MHz)')
plt.title('duffing nonlinearity vs resonant frequency')
plt.show()

plt.figure(figsize=(10,8))
plt.plot(f0_theory.ravel(),np.abs(duffing_theory.ravel())/1.0E6,'.')
plt.xlabel('f0')
plt.ylabel('abs(duffing) (MHz)')
plt.title('abs(duffing nonlinearity) vs resonant frequency')
plt.show()



