# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:15:28 2020

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt

def bistable_duffing_response(deltas,nin,f0,Tmix=0.03):
    imag_cutoff = 1.0E-20
    boltzmann_factor = np.exp(((-6.63E-34)*f0)/((1.38E-23)*Tmix))
    response = np.zeros_like(deltas)
    for ii in range(len(deltas)):
        delta = deltas[ii]
        coeffs = np.array([1,-2*delta,(delta**2)+(1/4),-nin])
        roots = np.roots(coeffs)
        im_part = np.imag(roots)
        real_roots = np.real(roots[np.where(np.abs(im_part)<imag_cutoff)])
        if len(real_roots) == 1:
            response[ii] = real_roots[0]
        elif len(real_roots) == 2:
            print('ERROR: FOUND ONLY 2 ROOTS!!!')
        else:
            sol1 = real_roots[0]
            sol2 = real_roots[2]
            response[ii] = (((sol1*(boltzmann_factor**sol1))+(sol2*(boltzmann_factor**sol2))))/((boltzmann_factor**sol1)+(boltzmann_factor**sol2))
    return response

n_deltas = 10000
n_deltas = 5001
delta_span = 5.0
delta0 = np.sqrt(3.0)/2.0
deltas = np.linspace(-delta_span, delta_span, n_deltas)+delta0
nin = (np.sqrt(3.0)/9.0)+3.0
f0 = 5.75E9
response = bistable_duffing_response(deltas,nin,f0)

plt.plot(deltas,response,'.')
plt.show()