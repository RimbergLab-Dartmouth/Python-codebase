# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:09:49 2020

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
ngs = np.linspace(-0.65,0.65,50)
phis = np.linspace(-np.pi,np.pi,50)
fcpt = bandfitr.fCPT(ngs,phis,fc,fj,n_charges=n_charges)

fig = plotr.plot_band_colormap(phis,ngs,fcpt,title='fcpt')
plt.show()
