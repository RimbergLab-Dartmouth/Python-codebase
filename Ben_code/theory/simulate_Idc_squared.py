# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:58:51 2020

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
n_points = 50
ngs = np.linspace(-0.65,0.65,n_points)
phis = np.linspace(-np.pi,np.pi,n_points)
cos_dphis = bandfitr.avg_cos_dphi(ngs,phis,fc,fj,n_charges=n_charges)

#fig = plotr.plot_band_colormap(phis,ngs,cos_dphis,title='cos(dphi) Theory')
#plt.show()

phis_array = np.outer(phis,np.ones(len(ngs)))

Idc = np.sin(phis_array)*cos_dphis
fig = plotr.plot_band_colormap(phis,ngs,Idc**2.0,title='Idc**2 Theory')
plt.show()

fig = plotr.plot_band_colormap(phis,ngs,np.abs(Idc),title='abs(Idc) Theory')
plt.show()

d2f = bandfitr.dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
f0_theory = f0_bare + ((phizp**2)*d2f)


"""
kappa = (Idc**2)/(f0_theory)
fig = plotr.plot_band_colormap(phis,ngs,kappa,title='Idc**2 Theory')
plt.show()
"""

dphiext2 = 1.0/(np.abs(Idc)*(np.cos(phis_array)**2))
fig = plotr.plot_band_colormap(phis,ngs,dphiext2,title='expectation(dphiext**2) Theory')
plt.show()


plt.plot(f0_theory.ravel(),(Idc**2.0).ravel(),'.')
plt.xlabel('f0')
plt.ylabel('Idc**2')
plt.show()
