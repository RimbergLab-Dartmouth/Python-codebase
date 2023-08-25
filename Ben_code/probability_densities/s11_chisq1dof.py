# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:31:22 2020

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt
import plotting_routines as plotr
import scipy as sp

def s11_ideal(df,ktot,radius):
    xi = radius/(1-radius)
    kint = ktot/(1+xi)
    kext = (xi*ktot)/(1+xi)
    numerator = df-(1j*((kint-kext)/2))
    denominator = df-(1j*((kint+kext)/2))
    gamma = numerator/denominator
    return gamma

def s11_chisq1(df,ktot,radius,D):
    factor = (-1j*radius*ktot*np.sqrt(np.pi))/(2*D)
    argument = np.sqrt(((-2*df)+(1j*ktot))/(4*D))
    s11 = 1+((factor*sp.special.wofz(1j*argument))/argument)
    return s11

detuning_range = 20.0
n_detunings = 1000
dfs = np.linspace(-1*detuning_range,detuning_range,n_detunings)
radius = 0.8
D = 0.25

s11_chisq1s = s11_chisq1(dfs,1.0,radius,D)
s11s = s11_ideal(dfs,1.0,0.85*radius)

fig = plotr.plot_reflection_coefficient_fits(dfs,s11_chisq1s,s11s)
plt.show()




