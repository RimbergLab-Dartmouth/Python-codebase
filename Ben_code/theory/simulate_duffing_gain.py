# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:54:46 2020

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

plot_flag = True

def duffing_response(deltas,nin):
    bistability_flag = False
    imag_cutoff = 1.0E-20
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
            if not bistability_flag:
                print('Bistability Threshold Reached!')
                bistability_inds = []
                response2 = []
                response3 = []
            bistability_flag = True
            bistability_inds.append(ii)
            response[ii] = real_roots[0]
            response2.append(real_roots[1])
            response3.append(real_roots[2])
            
            
    if bistability_flag:
        return (response, response2, response3, bistability_inds)
    else:
        return response
    

flag = 0

n_deltas = 10001
delta_span = 5.0
delta0 = np.sqrt(3.0)/2.0
deltas = np.linspace(-delta_span, delta_span, n_deltas)+delta0

n_nins = 20
nin_max = 1.0*(np.sqrt(3.0)/9.0)
nins = np.linspace(0.01,nin_max,n_nins)

max_gains = np.zeros(n_nins,dtype=float)
max_detunings = np.zeros(n_nins,dtype=float)
ideal_detunings = np.zeros(n_nins,dtype=float)

for ii in range(n_nins):
    flag = 0
    nin = nins[ii]
    result = duffing_response(deltas,nin)
    if isinstance(result,type((0,0))):
        response, response2, response3, bistability_inds = result
        flag = 1
    else:
        response = result
    
    gain = np.abs((response - np.roll(response,1))[1:])/(deltas[1]-deltas[0])
    max_gains[ii] = np.amax(gain)
    max_detunings[ii] = deltas[np.argmax(gain)]
    
    if flag == 1:
        ideal_detunings[ii] = (deltas[bistability_inds[0]]+deltas[bistability_inds[-1]])/2.0
    else:
        ideal_detunings[ii] = deltas[np.argmax(gain)]
    
    if plot_flag:
        markersize = 0.25
        plt.plot(deltas,response,'.',markersize=markersize)
        plt.plot(deltas[np.argmax(gain)],response[np.argmax(gain)],'o')
        plt.xlabel('delta')
        plt.ylabel('response')
        plt.show()
        
        stop_flag = input('stop (y/n)?')
        if stop_flag=='y':
            sys.exit()
    
    
plt.plot(nins,max_gains,'.')
plt.xlabel('nin')
plt.ylabel('max gain')
plt.yscale('log')
plt.show()

plt.plot(max_gains,max_detunings,'.')
plt.xlabel('max gains')
plt.ylabel('max detuning')
plt.xscale('log')
plt.show()



x = nins
y = max_detunings
poly,covar = np.polyfit(x, y, 1, cov=True)
slope = poly[0]
intercept = poly[1]
errs = np.sqrt(np.diag(covar))
print('slope = '+str(slope)+' +/- '+str(errs[0]))
print('intercept = '+str(intercept)+' +/- '+str(errs[1]))
fit = intercept+(slope*x)

plt.plot(nins,max_detunings,'.')
plt.plot(nins,fit)
plt.xlabel('nin')
plt.ylabel('max detuning')
plt.show()


x = nins
y = ideal_detunings
poly,covar = np.polyfit(x, y, 1, cov=True)
slope = poly[0]
intercept = poly[1]
errs = np.sqrt(np.diag(covar))
print('slope = '+str(slope)+' +/- '+str(errs[0]))
print('intercept = '+str(intercept)+' +/- '+str(errs[1]))
fit = intercept+(slope*x)

plt.plot(nins,ideal_detunings,'.')
plt.plot(nins,fit)
plt.xlabel('nin')
plt.ylabel('ideal detuning')
plt.show()



