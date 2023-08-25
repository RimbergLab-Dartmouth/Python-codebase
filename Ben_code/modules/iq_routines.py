# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 13:56:06 2020

@author: Ben
"""

import numpy as np

def get_iq(time_axis, signal, if_freq, return_IQ = True):
    # retrieve quadrature data from signal vs time data
    # return_IQ = True: return in-phase and quadrature
    # return_IQ = False: return amplitude and phase
    iwave = np.cos(2*np.pi*if_freq*time_axis)
    qwave = np.sin(2*np.pi*if_freq*time_axis)
    
    samples = len(time_axis)
    samples_per_period = int(round(samples/((time_axis[-1]-time_axis[0])*if_freq)))
    
    iprod = iwave*signal
    qprod = qwave*signal
    inphase = 2*np.convolve(iprod,np.ones(samples_per_period)/samples_per_period,mode='valid')
    quadrature = 2*np.convolve(qprod,np.ones(samples_per_period)/samples_per_period,mode='valid')
    if return_IQ:
        return inphase, quadrature
    else:
        amplitude = np.sqrt((inphase**2)+(quadrature**2))
        phase = np.arctan(quadrature/inphase)
        return amplitude, phase