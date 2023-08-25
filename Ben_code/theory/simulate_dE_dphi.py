# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:44:42 2020

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
ng0 = 0.65
phi0 = np.pi
n_points = 21
ngs = np.linspace(-ng0,ng0,n_points)
phis = np.linspace(0,phi0,n_points)
df = bandfitr.dfCPT(ngs,phis,fc,fj,1,n_charges=n_charges)

fig = plotr.plot_band_colormap(phis,ngs,df,title='df/dphi Theory')
plt.show()
