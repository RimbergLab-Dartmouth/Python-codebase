# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:43:18 2020

@author: Ben
"""

import numpy as np

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
    
def get_ideal_detuning(nin,
                       n_deltas = 5001,
                       delta_span = 4.0,
                       delta0 = np.sqrt(3.0)/2.0):
    # Below threshold:
    #   finds detuning that maximizes dn/d(delta) 
    # Above threshold:
    #   finds center of bistable region
    deltas = np.linspace(-delta_span, delta_span, n_deltas)+delta0
    result = duffing_response(deltas,nin)
    if isinstance(result,type((0,0))):
        response, response2, response3, bistability_inds = result
        ideal_detuning = (deltas[bistability_inds[0]]+deltas[bistability_inds[-1]])/2.0
    else:
        ideal_detuning = deltas[np.argmax(np.abs((response - np.roll(response,1))[1:]))]
    return ideal_detuning







    