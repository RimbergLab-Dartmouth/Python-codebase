# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:45:00 2019

@author: Ben
"""

import numpy as np
import band_fitting_routines as bandfitr
import plotting_routines as plotr
import matplotlib.pyplot as plt

n_charges = 4
f0_bare = 5.757E9
fc = 54.8E9
fj = 14.9E9
phizp = 0.0878776
ngs = np.linspace(-0.8,0.8,50)
phis = np.linspace(-np.pi/2,np.pi/2,50)
d2f = bandfitr.dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
f0_theory = f0_bare + ((phizp**2)*d2f)
df0_dphi_theory = ((phizp**2)*bandfitr.dfCPT(ngs,phis,fc,fj,3,n_charges=n_charges))

fig = plotr.plot_band_colormap(phis,ngs,f0_theory,title='f0 Theory')
plt.show()

fig = plotr.plot_band_colormap(phis,ngs,df0_dphi_theory,title='df0/dphi Theory')
plt.show()

plt.plot(f0_theory.ravel(),np.abs(df0_dphi_theory.ravel()),'.')
plt.xlabel('f0')
plt.ylabel('df0/dphi')
plt.show()