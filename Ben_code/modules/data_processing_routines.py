# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:14:27 2019

@author: Ben
"""
import numpy as np
import math
from scipy.interpolate import interp1d

def get_slug_background(logmag_array, \
                        phase_array, \
                        freq_interval, \
                        exclusion_freq=10.0E6):
    n_fluxes, n_gates, trace_points = logmag_array.shape
    
    logmag_background = np.zeros(trace_points)
    phase_background = np.zeros(trace_points)
    averaging_count = np.zeros(trace_points)
    
    exclusion_inds = math.ceil(exclusion_freq/freq_interval)
    simple_logmag_background = np.mean(np.mean(logmag_array,axis=0),axis=0)

    for ii in range(n_fluxes):
        for jj in range(n_gates):
            logmag = np.copy(logmag_array[ii,jj])
            phase = np.copy(phase_array[ii,jj])
            
            without_background = logmag - simple_logmag_background
            min_ind = np.argmin(without_background)
    
            logmag[min_ind-exclusion_inds:min_ind+exclusion_inds]=0.0
            logmag_background += logmag
            
            phase[min_ind-exclusion_inds:min_ind+exclusion_inds]=0.0
            phase_background += phase
            
            averaging_count += 1
            averaging_count[min_ind-exclusion_inds:min_ind+exclusion_inds] -= 1
            
    logmag_background = logmag_background/averaging_count
    phase_background = phase_background/averaging_count
    return logmag_background, phase_background

def get_fast_slug_background(logmag_array, \
                        phase_array, \
                        freq_interval, \
                        exclusion_freq=10.0E6):
    n_fluxes, trace_points = logmag_array.shape
    
    logmag_background = np.zeros(trace_points)
    phase_background = np.zeros(trace_points)
    averaging_count = np.zeros(trace_points)
    
    exclusion_inds = math.ceil(exclusion_freq/freq_interval)
    simple_logmag_background = np.mean(logmag_array,axis=0)

    for ii in range(n_fluxes):
            logmag = np.copy(logmag_array[ii])
            phase = np.copy(phase_array[ii])
            
            without_background = logmag - simple_logmag_background
            min_ind = np.argmin(without_background)
    
            logmag[min_ind-exclusion_inds:min_ind+exclusion_inds]=0.0
            logmag_background += logmag
            
            phase[min_ind-exclusion_inds:min_ind+exclusion_inds]=0.0
            phase_background += phase
            
            averaging_count += 1
            averaging_count[min_ind-exclusion_inds:min_ind+exclusion_inds] -= 1
            
    logmag_background = logmag_background/averaging_count
    phase_background = phase_background/averaging_count
    return logmag_background, phase_background

def get_fast_slug_logmag_background(logmag_array, \
                        freq_interval, \
                        exclusion_freq=10.0E6):
    n_fluxes, trace_points = logmag_array.shape
    logmag_background = np.zeros(trace_points)
    averaging_count = np.zeros(trace_points)
    
    exclusion_inds = math.ceil(exclusion_freq/freq_interval)
    simple_logmag_background = np.mean(logmag_array,axis=0)

    for ii in range(n_fluxes):
            logmag = np.copy(logmag_array[ii])
            
            without_background = logmag - simple_logmag_background
            min_ind = np.argmin(without_background)
    
            logmag[min_ind-exclusion_inds:min_ind+exclusion_inds]=0.0
            logmag_background += logmag
            
            averaging_count += 1
            averaging_count[min_ind-exclusion_inds:min_ind+exclusion_inds] -= 1
            
    logmag_background = logmag_background/averaging_count
    return logmag_background


def get_avg_biases(flux_array, gate_array):
    avg_fluxes = np.mean(flux_array,axis=1)
    avg_gates = np.mean(gate_array,axis=0)
    return avg_fluxes,avg_gates

def get_s11_arrays(logmag_array, \
                   phase_array, \
                   freq_array, \
                   logmag_background, \
                   phase_background, \
                   background_freqs):
    
    shape = logmag_array.shape
    if len(shape)==4:
        n_fluxes,n_gates,n_powers,n_freqs = shape
    elif len(shape)==3:
        n_fluxes,n_gates,n_freqs = shape
    else:
        print('UNEXPECTED ARRAY SHAPE!!')
        
    s11_logmag_array = np.zeros_like(logmag_array)
    s11_phase_array = np.zeros_like(phase_array)
    
    interp_logmag = interp1d(background_freqs,logmag_background, kind='cubic')
    interp_phase = interp1d(background_freqs,phase_background, kind='cubic')
    
    for ii in range(n_fluxes):
        for jj in range(n_gates):
            freq_array[ii,jj][np.where(freq_array[ii,jj]>background_freqs[-1])] = background_freqs[-1]
            freq_array[ii,jj][np.where(freq_array[ii,jj]<background_freqs[0])] = background_freqs[0]
            this_logmag_bg = interp_logmag(freq_array[ii,jj])
            this_phase_bg = interp_phase(freq_array[ii,jj])
            if len(shape) == 3:
                s11_logmag_array[ii,jj] = logmag_array[ii,jj]-this_logmag_bg
                s11_phase_array[ii,jj] = phase_array[ii,jj]-this_phase_bg
            elif len(shape) == 4:
                for kk in range(n_powers):
                    s11_logmag_array[ii,jj,kk] = logmag_array[ii,jj,kk]-this_logmag_bg
                    s11_phase_array[ii,jj,kk] = phase_array[ii,jj,kk]-this_phase_bg
    
    return s11_logmag_array,s11_phase_array
    
    
    
    
    
    
    
    
    
    
    