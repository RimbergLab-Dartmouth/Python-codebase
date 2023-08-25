# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:53:52 2019

@author: Ben
"""

import numpy as np
import testing_reflection_fitting as testreflectr
import reflection_fitting_routines as reflectr
import plotting_routines as plotr
import matplotlib.pyplot as plt
import time


h = 6.626*(10**-34)

f0 = 5.7374E9
f = f0+np.linspace(-15.0E6,15.0E6,500)
g_int = 0.05E6
g_ext = 1.3E6
theta = 0
sigma = 0.7E6
duff = 0.5E6
Pin = 4*h*f0*(g_int+g_ext)
print(Pin)

t0 = time.time()
gamma_theory = testreflectr.s11_duffing_theory3(f,f0,g_int,g_ext,sigma,duff,Pin, \
                                        x_samples =    100, \
                                        sigma_range = 4.0, \
                                        bistability_method='hysteresis')
t1 = time.time()
print('total time: '+str(t1-t0)+' seconds')

fit_result = reflectr.make_s11_duffing(f0, g_int, g_ext, theta, sigma,\
                                       power_units='W')(f,duff,Pin)
print('len of make s11 result: '+str(len(fit_result)))
real_part = fit_result[0:int(len(fit_result)/2)]
im_part = fit_result[int(len(fit_result)/2):]
gamma_fit = real_part + (1.0j*im_part)

fig = plotr.plot_reflection_coefficient(f,gamma_theory)
plt.show()

fig = plotr.plot_reflection_coefficient(f,gamma_fit)
plt.show()

