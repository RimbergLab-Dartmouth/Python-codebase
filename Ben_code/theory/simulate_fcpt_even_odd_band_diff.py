# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:26:26 2020

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

ng0 = 0.8
n_points = 50
ngs = np.linspace(-ng0,ng0,n_points)
phis = np.linspace(-np.pi,np.pi,n_points)

fcpt_even = bandfitr.fCPT(ngs,phis,fc,fj,n_charges=n_charges)
fcpt_odd = bandfitr.fCPT(ngs+1,phis,fc,fj,n_charges=n_charges)
fcpt_diff = fcpt_even-fcpt_odd

fig = plotr.plot_band_colormap(phis,ngs,fcpt_even,title='fcpt even')
plt.show()

fig = plotr.plot_band_colormap(phis,ngs,fcpt_odd,title='fcpt odd')
plt.show()

fig = plotr.plot_band_colormap(phis,ngs,np.abs(fcpt_diff),title='abs(fcpt diff)')
plt.show()
