# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:47:59 2020

@author: Ben
"""

import numpy as np
import band_fitting_routines as bandfitr
import plotting_routines as plotr
import matplotlib.pyplot as plt
from scipy import special

n_charges = 4
f0_bare = 5.757E9
fc = 54.8E9
fj = 14.9E9
phizp = 0.0878776
n_points = 50
ngs = np.linspace(-0.65,0.65,n_points)
phis = np.linspace(-np.pi,np.pi,n_points)
var_cos_dphis = bandfitr.var_cos_dphi(ngs,phis,fc,fj,n_charges=n_charges)

#fig = plotr.plot_band_colormap(phis,ngs,cos_dphis,title='cos(dphi) Theory')
#plt.show()

phis_array = np.outer(phis,np.ones(len(ngs)))

phizpf = 0.088
n_photons = 0.5
var_Idc = (1/2)*(1-special.jv(0,2*phizpf*np.sqrt(n_photons)))*var_cos_dphis
fig = plotr.plot_band_colormap(phis,ngs,var_Idc**2.0,title='Idc**2 Theory')
plt.show()


d2f = bandfitr.dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
f0_theory = f0_bare + ((phizp**2)*d2f)


"""
kappa = (Idc**2)/(f0_theory)
fig = plotr.plot_band_colormap(phis,ngs,kappa,title='Idc**2 Theory')
plt.show()
"""

dphiext2 = 1.0/(np.abs(var_Idc)*(np.cos(phis_array)**2))
fig = plotr.plot_band_colormap(phis,ngs,dphiext2,title='expectation(dphiext**2) Theory')
plt.show()

