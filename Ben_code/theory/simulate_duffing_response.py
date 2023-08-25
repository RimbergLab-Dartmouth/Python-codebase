# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:31:30 2020

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt

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
            """
            print('roots:')
            print(roots)
            print('real roots:')
            print(real_roots)
            print('imag part:')
            print(im_part)
            """
            
            """
            a = coeffs[0]
            b = coeffs[1]
            c = coeffs[2]
            d = coeffs[3]
            discrim = (18*a*b*c*d)-(4*(b**3)*d)+((b*c)**2)-(4*a*(c**3))-(27*((a*d)**2))
            print('Discriminant = '+str(discrim))
            """
            
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

n_deltas = 10000
n_deltas = 5001
delta_span = 20.0
delta0 = np.sqrt(3.0)/2.0
deltas = np.linspace(-delta_span, delta_span, n_deltas)+delta0
nin = 1.0/8.0
result = duffing_response(deltas,nin)
if isinstance(result,type((0,0))):
    response, response2, response3, bistability_inds = result
    flag = 1
else:
    response = result


markersize = 0.5
if flag == 0:
    plt.plot(deltas,response,'.',markersize=markersize)
    plt.xlabel('Detuning')
    plt.ylabel('Duffing Response')
    plt.show()

elif flag == 1:
    bistable_deltas = deltas[bistability_inds]

    plt.plot(deltas,response,'.',markersize=markersize)
    plt.plot(bistable_deltas,response2,'.',markersize=markersize)
    plt.plot(bistable_deltas,response3,'.',markersize=markersize)
    plt.xlabel('Detuning')
    plt.ylabel('Duffing Response')
    plt.show()






