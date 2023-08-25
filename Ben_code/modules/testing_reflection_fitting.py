# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 23:14:17 2019

@author: Ben
"""

import scipy as sp
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import time
import lmfit
import plotting_routines as plotr
import sys

def s11_noisy(f,f0,ktot,radius,sigma):
    df = f-f0
    factor = (-1*radius*np.pi*ktot)/(np.sqrt(2*np.pi*(sigma**2)))
    argument = ((-2*df)+(1j*ktot))/(2*np.sqrt(2)*sigma)
    s11 = 1+(factor*sp.special.wofz(argument))
    return s11

def s11_noisy_mismatched(f,f0,ktot,radius,sigma,theta):
    df = f-f0
    factor = (-1*radius*np.pi*ktot)/(np.sqrt(2*np.pi*(sigma**2)))
    argument = ((-2*df)+(1j*ktot))/(2*np.sqrt(2)*sigma)
    s11 = 1+(factor*sp.special.wofz(argument))
    s11 = ((s11-1.0)*np.exp(-1j*theta))+1.0
    return s11

def s11_noisy_mismatched_int_ext(f,f0,kint,kext,sigma,theta):
    df = f-f0
    ktot = kint+kext
    radius = kext/ktot
    factor = (-1*radius*np.pi*ktot)/(np.sqrt(2*np.pi*(sigma**2)))
    argument = ((-2*df)+(1j*ktot))/(2*np.sqrt(2)*sigma)
    s11 = 1+(factor*sp.special.wofz(argument))
    s11 = ((s11-1.0)*np.exp(-1j*theta))+1.0
    return s11

def s11_noisy_mismatch_param(f,f0,kint,kext,sigma,imagkext):
    df = f-f0
    kexttotal = kext + (1j*imagkext)
    ktot = kint+kexttotal
    radius = kexttotal/ktot
    factor = (-1*radius*np.pi*ktot)/(np.sqrt(2*np.pi*(sigma**2)))
    argument = ((-2*df)+(1j*ktot))/(2*np.sqrt(2)*sigma)
    s11 = 1+(factor*sp.special.wofz(argument))
    return s11

def s11_ideal(f,f0,kint,kext):
    # complex reflection coefficient
    df = f-f0
    numerator = df-(1j*((kint-kext)/2))
    denominator = df-(1j*((kint+kext)/2))
    gamma = numerator/denominator
    return gamma

def s11_rotated(f,f0,kint,kext,theta):
    # complex reflection coefficient
    df = f-f0
    numerator = df-(1j*((kint-kext)/2))
    denominator = df-(1j*((kint+kext)/2))
    gamma = numerator/denominator
    gamma = ((gamma-1.0)*np.exp(-1j*theta))+1.0
    return gamma

def new_s11_model(f,f0,ktot,radius,sigma):
    df = f-f0
    factor = (-1*radius*np.pi*ktot)/(np.sqrt(2*np.pi*(sigma**2)))
    argument = ((2*df)-(1j*ktot))/(2*np.sqrt(2)*sigma)
    term1 = np.exp(-1*(argument**2))
    term2 = ((-2j)/np.sqrt(np.pi))*sp.special.dawsn(argument)
    s11 = 1+(factor*(term1+term2))
    return s11

def s11_model(f,f0,g_int,g_ext,sigma):
    #f = f.astype(np.float64)
    df = f-f0
    term = (df-(1j*((g_int+g_ext)/2)))/(sigma*np.sqrt(2))
    #term = term.astype(np.complex128)
    term1 = ((1j*np.pi*g_ext)/(np.sqrt(2*np.pi*(sigma**2))))
    term2 = 1j*np.exp(-1*(term**2))
    term3 = (2/np.sqrt(np.pi))*sp.special.dawsn(term)
    gamma = 1 + (term1*(term2+term3))
    return gamma

def make_s11_fixed_damping(gamma_data):
    def s11_fixed_damping(freqs_array, g_int0, g_ext0):
        plot_flag = False
        model = lmfit.Model(s11_model)
        params = model.make_params()
        params['g_int'].value = g_int0
        params['g_int'].vary = False
        params['g_ext'].value = g_ext0
        params['g_ext'].vary = False
        params['sigma'].value = 1.0E6
        params['sigma'].min = 0.0
        params['sigma'].max = 5.0E6
        n_fluxes,n_gates,n_freqs = freqs_array.shape
        gamma_array = np.zeros((n_fluxes,n_gates,n_freqs),dtype=np.complex128)
        for ii in range(n_fluxes):
            #print('flux '+str(ii))
            for jj in range(n_gates):
                #print('gate '+str(jj))
                freqs = freqs_array[ii,jj]
                gamma = gamma_data[ii,jj]
                params['f0'].value = freqs[round((len(freqs)-1)/2)]
                params['f0'].min = freqs[0]
                params['f0'].max = freqs[-1]
                result = model.fit(gamma,params,f=freqs)
                gamma_array[ii,jj] = result.best_fit
                if plot_flag:
                    print('g int = '+str(g_int0))
                    print('g ext = '+str(g_ext0))
                    print(result.fit_report())
                    print('ii,jj = '+str(ii)+','+str(jj))
                    fig = plotr.plot_reflection_coefficient_fits(freqs,gamma,result.best_fit)
                    plt.show()
                    stop_flag = input('stop?')
                    if stop_flag:
                        sys.exit()
        print('g int = '+str(g_int0))
        print('g ext = '+str(g_ext0))
        print('\n')
        return gamma_array
    return s11_fixed_damping

def make_s11_fixed_damping_residual(gamma_data,freqs_array):
    def s11_fixed_damping(fixed_damping_params):
        v = fixed_damping_params.valuesdict()
        g_int0 = v['g_int0']
        g_ext0 = v['g_ext0']
        plot_flag = False
        model = lmfit.Model(s11_model)
        params = model.make_params()
        params['g_int'].value = g_int0
        params['g_int'].vary = False
        params['g_ext'].value = g_ext0
        params['g_ext'].vary = False
        params['sigma'].value = 1.0E6
        params['sigma'].min = 0.0
        params['sigma'].max = 5.0E6
        n_fluxes,n_gates,n_freqs = freqs_array.shape
        gamma_array = np.zeros((n_fluxes,n_gates,n_freqs),dtype=np.complex128)
        for ii in range(n_fluxes):
            #print('flux '+str(ii))
            for jj in range(n_gates):
                #print('gate '+str(jj))
                freqs = freqs_array[ii,jj]
                gamma = gamma_data[ii,jj]
                params['f0'].value = freqs[round((len(freqs)-1)/2)]
                params['f0'].min = freqs[0]
                params['f0'].max = freqs[-1]
                result = model.fit(gamma,params,f=freqs)
                gamma_array[ii,jj] = result.best_fit
                if plot_flag:
                    print('g int = '+str(g_int0))
                    print('g ext = '+str(g_ext0))
                    print(result.fit_report())
                    print('ii,jj = '+str(ii)+','+str(jj))
                    fig = plotr.plot_reflection_coefficient_fits(freqs,gamma,result.best_fit)
                    plt.show()
                    stop_flag = input('stop?')
                    if stop_flag:
                        sys.exit()
        print('g int = '+str(g_int0))
        print('g ext = '+str(g_ext0))
        print('\n')
        #residual = np.sum(np.abs(gamma_data - gamma_array)**2)
        return np.abs(gamma_data - gamma_array)
    return s11_fixed_damping