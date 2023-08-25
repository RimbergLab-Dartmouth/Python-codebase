# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:26:26 2020

@author: Ben
"""

import numpy as np
import band_fitting_routines as bandfitr
import plotting_routines as plotr
import matplotlib.pyplot as plt

n_charges = 5
h = 4.14E-15 #plancks constant in eV*s
f0_bare = 5.757E9
Ec = h*54.8E9
Ej = h*14.9E9
phizp = 0.0878776
Tmix = 0.03
kb = 8.6E-5 #boltzmann constant in eV/K

ng0 = 1.0
n_points = 50
ngs = np.linspace(-ng0,ng0,n_points)
phis = np.linspace(-np.pi,np.pi,n_points)

Ecpt_even = bandfitr.fCPT(ngs,phis,Ec,Ej,n_charges=n_charges)/(kb*Tmix)
Ecpt_odd = bandfitr.fCPT(ngs+1,phis,Ec,Ej,n_charges=n_charges)/(kb*Tmix)
Ecpt_diff = Ecpt_odd-Ecpt_even

gap_energy_leads = 3.4E-4/(kb*Tmix)
gap_energy_island = 4.3E-4/(kb*Tmix)
gap_energy_diff = gap_energy_leads - gap_energy_island

fig = plotr.plot_band_colormap(phis,ngs,Ecpt_even,title='Ecpt even')
plt.show()

fig = plotr.plot_band_colormap(phis,ngs,Ecpt_odd,title='Ecpt odd')
plt.show()

fig = plotr.plot_band_colormap(phis,ngs,np.abs(Ecpt_diff),title='abs(Ecpt diff)')
plt.show()

quasiparticle_energy = np.abs(Ecpt_diff-gap_energy_diff)
fig = plotr.plot_band_colormap(phis,ngs,quasiparticle_energy,title='quasiparticle energy')
plt.show()
