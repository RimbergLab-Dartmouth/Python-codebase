# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:56:48 2019

@author: Ben
"""

import math
import scipy as scp
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import band_fitting_routines as bandfitr

def make_f0ccpt(n_charges=2):
    def f0ccpt(biases,fc,fj,f0_bare):
        phizp = 0.0878776
        n_good_fluxes = int(biases[0])
        phis = biases[1:n_good_fluxes+1]
        ngs = biases[n_good_fluxes+1:]
        d2f = fj*bandfitr.dfCPT(ngs,phis,fc/fj,1.0,2,n_charges=n_charges)
        f0 = f0_bare + ((phizp**2)*d2f)
        return f0
    return f0ccpt

def make_f0ccpt_with_kerr(n_charges=2):
    def f0ccpt(biases,fc,fj,f0_bare):
        phizp = 0.0878776
        n_good_fluxes = int(biases[0])
        phis = biases[1:n_good_fluxes+1]
        ngs = biases[n_good_fluxes+1:]
        d2f = fj*bandfitr.dfCPT(ngs,phis,fc/fj,1.0,2,n_charges=n_charges)
        d4f = fj*bandfitr.dfCPT(ngs,phis,fc/fj,1.0,4,n_charges=n_charges)
        f0 = f0_bare + ((phizp**2)*d2f) + ((0.25*(phizp**4))*d4f)
        return f0
    return f0ccpt

def make_f0ccpt_fcn(n_charges=2):
    def f0ccpt(bias_voltages,fc,fj,f0_bare,Vg0,Vphi0,Cg,Lphi):
        n_good_fluxes = int(bias_voltages[0])
        flux_voltages = bias_voltages[1:n_good_fluxes+1]
        gate_voltages = bias_voltages[n_good_fluxes+1:]
        phizp = 0.0878776
        phis = Lphi*(flux_voltages-Vphi0)
        ngs = Cg*(gate_voltages-Vg0)
        d2f = bandfitr.dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
        d4f = bandfitr.dfCPT(ngs,phis,fc,fj,4,n_charges=n_charges)
        f0 = f0_bare + ((phizp**2)*d2f) + ((0.25*(phizp**4))*d4f)
        #print('shape of f0 = '+str(f0.shape))
        return f0
    return f0ccpt

def make_f0ccpt_fcn_simple(n_charges=2):
    def f0ccpt(bias_voltages,fc,fj,f0_bare):
        phizp = 0.0878776
        n_good_fluxes = int(bias_voltages[0])
        phis = bias_voltages[1:n_good_fluxes+1]
        ngs = bias_voltages[n_good_fluxes+1:]
        d2f = bandfitr.dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
        d4f = bandfitr.dfCPT(ngs,phis,fc,fj,4,n_charges=n_charges)
        f0 = f0_bare + ((phizp**2)*d2f) + ((0.25*(phizp**4))*d4f)
        #print('shape of f0 = '+str(f0.shape))
        return f0
    return f0ccpt

def make_f0ccpt_noise_var(fc=51.1E9,fj=14.4E9,n_charges=2):
    def f0ccpt_noise_var(biases,sigma_ng,sigma_phi):
        n_good_fluxes = int(biases[0])
        phis = biases[1:n_good_fluxes+1]
        ngs = biases[n_good_fluxes+1:]
        phizp = 0.088
        dfdphi3 = (phizp**2)*bandfitr.dfCPT(ngs,phis,fc,fj,3,n_charges=n_charges)
        dfdphi2dng = (phizp**2)*bandfitr.dfCPT_dphi_dng(ngs,phis,fc,fj,2,1,n_charges=n_charges)
        result = np.sqrt(((sigma_phi*dfdphi3)**2)+((sigma_ng*dfdphi2dng)**2))
        return result
    return f0ccpt_noise_var

def make_f0ccpt_noise_var_with_shot(fc=51.1E9,fj=14.4E9,n_charges=2):
    def f0ccpt_noise_var(biases,sigma_ng,sigma_phi,n_photons):
        n_good_fluxes = int(biases[0])
        phis = biases[1:n_good_fluxes+1]
        ngs = biases[n_good_fluxes+1:]
        phizp = 0.088
        dfdphi3 = (phizp**2)*bandfitr.dfCPT(ngs,phis,fc,fj,3,n_charges=n_charges)
        dfdphi2dng = (phizp**2)*bandfitr.dfCPT_dphi_dng(ngs,phis,fc,fj,2,1,n_charges=n_charges)
        duffing = ((phizp**4)/2)*bandfitr.dfCPT(ngs,phis,fc,fj,4,n_charges=n_charges)
        result = np.sqrt(((sigma_phi*dfdphi3)**2)+((sigma_ng*dfdphi2dng)**2)+((n_photons*duffing)**2))
        return result
    return f0ccpt_noise_var

def make_f0ccpt_noise_var_with_flux_shot(fc=51.1E9,fj=14.4E9,n_charges=2):
    def f0ccpt_noise_var(biases,sigma_ng,sigma_phi,shot_coeff,voltage_offset):
        n_good_fluxes = int(biases[0])
        n_good_gates = int(biases[1])
        phis = biases[2:n_good_fluxes+2]
        ngs = biases[n_good_fluxes+2:n_good_fluxes+n_good_gates+2]
        fluxes = biases[n_good_fluxes+n_good_gates+2:]-voltage_offset
        phizp = 0.088
        dfdphi3 = (phizp**2)*bandfitr.dfCPT(ngs,phis,fc,fj,3,n_charges=n_charges)
        dfdphi2dng = (phizp**2)*bandfitr.dfCPT_dphi_dng(ngs,phis,fc,fj,2,1,n_charges=n_charges)
        shot_noise = np.abs(shot_coeff)*np.outer(np.abs(fluxes),np.ones(n_good_gates))
        result = np.sqrt(((sigma_phi*dfdphi3)**2)+((sigma_ng*dfdphi2dng)**2)+(shot_noise*(dfdphi3**2)))
        return result
    return f0ccpt_noise_var
        
        

