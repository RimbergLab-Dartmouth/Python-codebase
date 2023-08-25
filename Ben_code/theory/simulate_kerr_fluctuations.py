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
fc = 54.1E9
fj = 14.8E9
phizp = 0.0878776
ng0 = 0.6
phi0 = np.pi/2
n_points = 30
ngs = np.linspace(-ng0,ng0,n_points)
phis = np.linspace(-phi0,phi0,n_points)
d2f = bandfitr.dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
f0_theory = f0_bare + ((phizp**2)*d2f)

fig = plotr.plot_band_colormap(phis,ngs,f0_theory,title='f0 Theory')
plt.show()


kerr = ((phizp**4)/2)*bandfitr.dfCPT(ngs,phis,fc,fj,4,n_charges=n_charges)

fig = plotr.plot_band_colormap(phis,ngs,kerr,title='kerr nonlinearity')
plt.show()

dkerr_dphi = ((phizp**4)/2)*bandfitr.dfCPT(ngs,phis,fc,fj,5,n_charges=n_charges)
dkerr_dng = ((phizp**4)/2)*bandfitr.dfCPT_dphi_dng(ngs,phis,fc,fj,4,1,n_charges=n_charges)

sigma_phi = 8.0E-3
sigma_ng = 6.0E-3

dkerr = np.sqrt(((dkerr_dphi*sigma_phi)**2)+((dkerr_dng*sigma_ng)**2))

fig = plotr.plot_band_colormap(phis,ngs,dkerr,title='kerr nonlinearity fluctuations')
plt.show()
