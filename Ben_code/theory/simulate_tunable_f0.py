# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:43:30 2019

@author: Ben
"""

import numpy as np
import band_fitting_routines as bandfitr
import plotting_routines as plotr
import matplotlib.pyplot as plt

n_charges = 4
f0_bare = 5.757E9
fc = 243.0E9
fj = 31.0E9
phizp = 0.0878776
ng0 = 0.65
phi0 = np.pi
n_points = 21
ngs = np.linspace(-ng0,ng0,n_points)
phis = np.linspace(0,phi0,n_points)
d2f = bandfitr.dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
f0_theory = f0_bare + ((phizp**2)*d2f)

fig = plotr.plot_band_colormap(phis,ngs,f0_theory,title='f0 Theory')
plt.show()
