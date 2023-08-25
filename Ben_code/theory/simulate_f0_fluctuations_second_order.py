# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:35:15 2020

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
ng0 = 0.65
phi0 = np.pi/2.0
ngs = np.linspace(-ng0,ng0,n_points)
phis = np.linspace(-phi0,phi0,n_points)
d2f = bandfitr.dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
f0_theory = f0_bare + ((phizp**2)*d2f)

sigma_ng = 6.7E-3
sigma_phi = 8.8E-3

dfdphi3 = np.abs((phizp**2)*bandfitr.dfCPT(ngs,phis,fc,fj,3,n_charges=n_charges))
dfdphi2dng = np.abs((phizp**2)*bandfitr.dfCPT_dphi_dng(ngs,phis,fc,fj,2,1,n_charges=n_charges))

dfdphi4 = np.abs((phizp**2)*bandfitr.dfCPT(ngs,phis,fc,fj,4,n_charges=n_charges))
dfdphi2dng2 = np.abs((phizp**2)*bandfitr.dfCPT_dphi_dng(ngs,phis,fc,fj,2,2,n_charges=n_charges))

fig = plotr.plot_band_colormap(phis,ngs,f0_theory,title='f0 Theory')
plt.show()

fig = plotr.plot_band_colormap(phis,ngs,(sigma_phi)*dfdphi3,title='d3f0/dphi3*sigmaphi')
plt.show()

fig = plotr.plot_band_colormap(phis,ngs,(sigma_ng)*dfdphi2dng,title='d3f0/dphi2dng*sigmang')
plt.show()

fig = plotr.plot_band_colormap(phis,ngs,0.5*(sigma_phi**2)*dfdphi4,title='d4f0/dphi4*sigmaphi**2')
plt.show()

fig = plotr.plot_band_colormap(phis,ngs,0.5*(sigma_ng**2)*dfdphi2dng2,title='d4f0/dphi2dng2*sigmang**2')
plt.show()

