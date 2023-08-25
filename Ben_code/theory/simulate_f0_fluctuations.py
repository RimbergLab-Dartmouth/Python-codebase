# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:20:47 2020

@author: Ben
"""

import numpy as np
import band_fitting_routines as bandfitr
import plotting_routines as plotr
import matplotlib.pyplot as plt
import testing_band_fitting as testbandfitr

n_charges = 2
f0_bare = 5.757E9
fc = 54.8E9
fj = 14.9E9
phizp = 0.0878776
n_points = 21
ng0 = 0.6
phi0 = np.pi/2
ngs = np.linspace(-ng0,ng0,n_points)
phis = np.linspace(-phi0,phi0,n_points)
d2f = bandfitr.dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
f0_theory = f0_bare + ((phizp**2)*d2f)
sigma_ng = 6.7E-3
sigma_phi = 8.8E-3
biases = np.hstack((len(phis),phis,ngs))
sigma_f0 = testbandfitr.make_f0ccpt_noise_var(fc=fc,fj=fj,n_charges=n_charges)(biases,sigma_ng,sigma_phi)

fig = plotr.plot_band_colormap(phis,ngs,f0_theory,title='f0 Theory')
plt.show()

fig = plotr.plot_band_colormap(phis,ngs,sigma_f0,title='Sigma f0 Theory')
plt.show()


df0dng = f0_theory


