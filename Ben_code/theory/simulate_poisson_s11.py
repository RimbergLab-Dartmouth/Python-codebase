# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:12:40 2020

@author: Ben
"""

import io_routines as io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import reflection_fitting_routines as reflectr
import testing_reflection_fitting as testreflectr
import plotting_routines as plotr
import lmfit
from scipy.special import exp1
import math


def s11_ideal(df,ktot,xi,theta):
    # complex reflection coefficient
    kint = ktot/(1+xi)
    kext = (ktot*xi)/(1+xi)
    numerator = df-(1j*((kint-kext)/2))
    denominator = df-(1j*((kint+kext)/2))
    gamma = numerator/denominator
    gamma = ((gamma-1)*np.exp(1j*theta))+1
    return gamma

def make_poisson_s11(n_terms=10):
    def s11_poisson(f,f0,xi,ktot,kerr,n_avg,theta):
        df = f-f0
        norm = 0.0
        result = np.zeros(len(df),dtype=complex)
        for kk in range(n_terms):
            factor = ((n_avg**kk)*np.exp(-1*n_avg))/(math.factorial(kk))
            print(factor)
            print((kerr*kk)/2)
            norm+=factor
            result += s11_ideal(df-((kerr*kk)),ktot,xi,theta)*factor
        #result = result/norm
        return result
    return s11_poisson

span = 10.0E6
f0 = 5.75E9
n_freqs = 10001
freqs = np.linspace(f0-(span/2),f0+(span/2),n_freqs)

ktot = 0.4E6
xi = 4.2
kerr = 0.6E6
n_avg = 5.0
theta = 0

s11 = make_poisson_s11()(freqs,f0,xi,ktot,kerr,n_avg,theta)
fig = plotr.plot_reflection_coefficient(freqs,s11)
plt.show()
