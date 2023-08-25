# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:42:15 2020

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt
import pdf_routines as pdf

def g(delta,xi):
    numerator = delta-(0.5j*((1-xi)/(1+xi)))
    denominator = delta-0.5j
    return numerator/denominator

def g_inv(s11,xi):
    delta = np.zeros_like(s11)
    delta = (1.0j/(2.0*(1.0+xi)))*(1.0-(xi*((1.0+s11)/(1.0-s11))))
    return delta

def g_prime(delta,xi):
    result = ((-1.0j*xi)/(1+xi))/((delta-0.5j)**2)
    return result

def Q(s11,xi,delta0,sigma):
    deltas = g_inv(s11,xi)
    result = pdf.gaussian_pdf(deltas,delta0,sigma)/np.abs(g_prime(deltas,xi))
    return result
    
xi = 5.0

n_points = 100
real_axis = np.linspace(-2.0,2.0,n_points)
imag_axis = np.linspace(-2.0,2.0,n_points)
s11s = np.outer(real_axis,np.ones(len(imag_axis)))+(1.0j*np.outer(np.ones(len(real_axis)),imag_axis))
Qs = Q(s11s,xi,0.0,1.0)

dreal = np.mean((real_axis-np.roll(real_axis,1))[1:])
real_edges = np.hstack((real_axis[0]-dreal,real_axis))+(dreal/2.0)
dimag = np.mean((imag_axis-np.roll(imag_axis,1))[1:])
imag_edges = np.hstack((imag_axis[0]-dimag,imag_axis))+(dimag/2.0)

scale = 7
fig, ax = plt.subplots(figsize=(scale,scale))
mesh = ax.pcolormesh(real_edges, imag_edges, Qs.T, cmap='inferno')
ax.set_title('S11 Histogram')
fig.colorbar(mesh, ax=ax)
ax.set_xlabel('Re(S11)')
ax.set_ylabel('Im(S11)')
ax.set_aspect('equal')
plt.show()
