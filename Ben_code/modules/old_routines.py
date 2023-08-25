# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:10:28 2019

@author: Ben
"""
import numpy as np
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fitting_routines import *
import scipy as scp
import scipy.optimize as opt
import math



def s11_duffing(f,f0,g_int,g_ext,duff,Nin):
    response = []
    for freq in f:
        coeffs = np.array([1.0, (f0-freq)/duff, (((freq-f0)**2)+(((g_int+g_ext)**2)/4))/(4*(duff**2)), (-g_ext*Nin)/(2*np.pi*(4*(duff**2)))])
        roots = np.roots(coeffs)
        real_roots = roots[~np.iscomplex(roots)]
        if len(real_roots) > 1:
            print('MORE THAN ONE REAL ROOT!')
        response.append(float(real_roots[0]))
    return np.array(response)


def s11_empirical(f,f0,g_int,g_ext,theta):
    # add an empirical parameter for the phase of the internal field
    gamma = s11(f,f0,g_int,g_ext)
    rotated_gamma = ((gamma-1)*np.exp(1j*theta)) + 1
    x = np.real(rotated_gamma)
    y = np.imag(rotated_gamma)
    return x,y


def s11_empirical_err(params,f,xdata,ydata):
    f0 = params[0]
    g_int = params[1]
    g_ext = params[2]
    theta = params[3]
    x,y = s11_empirical(f,f0,g_int,g_ext,theta)
    err = ((x-xdata)**2)+((y-ydata)**2)
    return err

def get_resonance_params_empirical(freqs, xdata, ydata, \
                         f0_guess, \
                         g_int_guess = 1.0E6, \
                         g_ext_guess = 1.0E6, \
                         theta_guess = 0.0, \
                         return_cost = False, \
                         bounds = True):
    param_guess = [f0_guess, g_int_guess, g_ext_guess, theta_guess]
    if bounds:
        q_bound = freqs[-1]-freqs[0]
        low_bounds = [freqs[0],0,0,-np.pi]
        high_bounds = [freqs[-1],q_bound,q_bound,np.pi]
        bounds_tuple = (low_bounds,high_bounds)
        result = opt.least_squares(s11_empirical_err,param_guess,args=(freqs,xdata,ydata),bounds=bounds_tuple)
    else:
        result = opt.least_squares(s11_empirical_err,param_guess,args=(freqs,xdata,ydata))
    if return_cost:
        return result.x, result.cost
    else:
        return result.x

def s11(f,f0,g_int,g_ext):
    # complex reflection coefficient
    df = f-f0
    numerator = (4*(df**2))+((g_int**2)-(g_ext**2))+(4j*g_ext*df)
    denominator = (4*(df**2))+((g_int + g_ext)**2)
    return numerator/denominator


def s11real(f,f0,g_int,g_ext):
    # x = real, y = imaginary
    gamma = s11(f,f0,g_int,g_ext)
    x = np.real(gamma)
    y = np.imag(gamma)
    return x,y

def s11logmag(f,f0,g_int,g_ext):
    x,y = s11real(f,f0,g_int,g_ext)
    mag = np.sqrt((x**2)+(y**2))
    logmag = 20*(np.log10(mag))
    return logmag


def s11err(params,f,xdata,ydata):
    f0 = params[0]
    g_int = params[1]
    g_ext = params[2]
    x,y = s11real(f,f0,g_int,g_ext)
    err = ((x-xdata)**2)+((y-ydata)**2)
    return err

def get_resonance_params(freqs, xdata, ydata, \
                         f0_guess, \
                         g_int_guess = 5.0E6, \
                         g_ext_guess = 5.0E6, \
                         return_cost = False, \
                         bounds = True):
    param_guess = [f0_guess, g_int_guess, g_ext_guess]
    if bounds:
        q_bound = freqs[-1]-freqs[0]
        low_bounds = [freqs[0],0,0]
        high_bounds = [freqs[-1],q_bound,q_bound]
        bounds_tuple = (low_bounds,high_bounds)
        result = opt.least_squares(s11err,param_guess,args=(freqs,xdata,ydata),bounds=bounds_tuple)
    else:
        result = opt.least_squares(s11err,param_guess,args=(freqs,xdata,ydata))
    if return_cost:
        return result.x, result.cost
    else:
        return result.x
    
def E0_points(n_charges, Vg, Iphi, Ec, Ej, Cg, Vg0, Lphi, Iphi0, ngcrit):
    # Vg, Iphi floats
    # number of charge states = 2*n_charges + 1
    ng = Cg*(Vg-Vg0)
    phi = Lphi*(Iphi-Iphi0)
    
    dim = 2*n_charges+1
    charges = np.linspace(-n_charges,n_charges,dim,dtype=int)
    offdiagvec = np.ones(dim-1)
    diagvec = np.ones(dim)    
    tunnel_coeff = -Ej*np.cos(phi)
    tunnel_mat = tunnel_coeff*(np.diag(offdiagvec,1)+np.diag(offdiagvec,-1))
    charge_coeff = 4*Ec
    for ii in range(dim):
        n = charges[ii]
        if ngcrit < ng%2 < 2-ngcrit:
            diagvec[ii] = (n-((ng+1)/2.0))**2.0
        else:
            diagvec[ii] = (n-(ng/2.0))**2.0
    charge_mat = charge_coeff*np.diag(diagvec)
    mat = charge_mat + tunnel_mat
    val = np.min(np.linalg.eigvalsh(mat))
    return val
    
def E0_arrays(n_charges, Vg, Iphi, Ec, Ej, Cg, Vg0, Lphi, Iphi0, ngcrit):
    # Vg, Iphi arrays
    # number of charge states = 2*n_charges + 1
    ng = Cg*(Vg-Vg0)
    phi = Lphi*(Iphi-Iphi0)
    
    n_gates = len(ng)
    n_fluxes = len(phi)
    vals = np.zeros((n_fluxes,n_gates))
    
    for ii in range(n_fluxes):
        this_phi = phi[ii]
        for jj in range(n_gates):
            this_ng = ng[jj]
            
            dim = 2*n_charges+1
            charges = np.linspace(-n_charges,n_charges,dim,dtype=int)
            offdiagvec = np.ones(dim-1)
            diagvec = np.ones(dim)    
            tunnel_coeff = -Ej*np.cos(this_phi)
            tunnel_mat = tunnel_coeff*(np.diag(offdiagvec,1)+np.diag(offdiagvec,-1))
            charge_coeff = 4*Ec
            for kk in range(dim):
                n = charges[kk]
                if ngcrit < this_ng%2 < 2-ngcrit:
                    diagvec[kk] = (n-((this_ng+1)/2.0))**2.0
                else:
                    diagvec[kk] = (n-(this_ng/2.0))**2.0
            charge_mat = charge_coeff*np.diag(diagvec)
            mat = charge_mat + tunnel_mat
            val = np.min(np.linalg.eigvalsh(mat))
            vals[ii,jj] = val
    return vals
    
def E0_ideal(n_charges, ng, phi, Ec, Ej):
    # ng, phi arrays
    n_gates = len(ng)
    n_fluxes = len(phi)
    vals = np.zeros((n_fluxes,n_gates))
    
    for ii in range(n_fluxes):
        this_phi = phi[ii]
        for jj in range(n_gates):
            this_ng = ng[jj]
            
            dim = 2*n_charges+1
            charges = np.linspace(-n_charges,n_charges,dim,dtype=int)
            offdiagvec = np.ones(dim-1)
            diagvec = np.ones(dim)    
            tunnel_coeff = -Ej*np.cos(this_phi)
            tunnel_mat = tunnel_coeff*(np.diag(offdiagvec,1)+np.diag(offdiagvec,-1))
            charge_coeff = 4*Ec
            for kk in range(dim):
                n = charges[kk]
                diagvec[kk] = (n-(this_ng/2.0))**2.0
            charge_mat = charge_coeff*np.diag(diagvec)
            mat = charge_mat + tunnel_mat
            val = np.min(np.linalg.eigvalsh(mat))
            vals[ii,jj] = val
    return vals

def omegaR(n_charges, Vg, Iphi, omega0, Ec, Ej, Cg, Vg0, Lphi, Iphi0, ngcrit):
    phizp = 0.0878776
    phi = Lphi*(Iphi-Iphi0)
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    E0 = E0_arrays(n_charges, Vg, Iphi, Ec, Ej, Cg, Vg0, Lphi, Iphi0, ngcrit)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi**2)
    omega = omega0 + 2*(phizp**2)*d2E0
    return omega
    
def omegaR_ideal(n_charges, ng, phi, Ec, Ej):
    phizp = 0.0878776
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    E0 = E0_ideal(n_charges, ng, phi, Ec, Ej)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi**2)
    omegaR = (phizp**2)*d2E0
    return omegaR[1:-1]

def fR_fitting_miles(params, inputs):
    # input fc, fj in Hz
    f0, fc, fj, Cg, Vg0, Lphi, Iphi0, ngcrit = params
    Iphi, Vg, n_charges = inputs
    phizp = 0.0878776
    phi = Lphi*(Iphi-Iphi0)
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    E0 = E0_arrays(n_charges, Vg, Iphi, fc, fj, Cg, Vg0, Lphi, Iphi0, ngcrit)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi**2)
    f = f0 + (phizp**2)*d2E0
    return f[1:-1]

def get_band_renormalization_params(Iphi, Vg, n_charges, f0_data, f0, fc, fj, Cg, Vg0, Lphi, Iphi0, ngcrit, return_covar = True):
    param_guesses = f0, fc, fj, Cg, Vg0, Lphi, Iphi0, ngcrit
    params, covar = opt.curve_fit(fR_fitting_miles,(Iphi,Vg,n_charges),f0_data,param_guesses)
    if return_covar:
        return params,covar
    else:
        return params
    
def fR_miles_err(params, Iphi, Vg, n_charges, f0_data):
    inputs = Iphi, Vg, n_charges
    fR = fR_fitting_miles(params, inputs)
    err = np.sum(fR - f0_data[1:-1])
    return err

def get_band_renormalization_params_lsq(Iphi, Vg, n_charges, f0_data, f0, fc, fj, Cg, Vg0, Lphi, Iphi0, ngcrit, return_cost = True):
    param_guesses = [f0, fc, fj, Cg, Vg0, Lphi, Iphi0, ngcrit]
    result = opt.least_squares(fR_miles_err,param_guesses,args=(Iphi,Vg,n_charges,f0_data))
    if return_cost:
        return result.x, result.cost
    else:
        return result.x
    
def fR_fitting(n_charges, Vg, Iphi, omega0, Ec, Ej, Cg, Vg0, Lphi, Iphi0, ngcrit):
    pass

def fR_miles_steepest_descent(params,param_steps,Iphi,Vg,n_charges,f0_data):
    inputs = Iphi, Vg, n_charges
    fR = fR_fitting_miles(params, inputs)
    dummy_params = tuple(params)
    for ii in range(len(params)):
        new_param = params[ii]+param_steps[ii]
    pass
    
    
def fR_simple(n_charges, Vg, Iphi, fc, fj):
    Vg0 = 0.01054054
    Cg = 40.65
    Iphi0 = -1.024E-5
    Lphi = 69804
    

def s11_empirical(f,f0,g_int,g_ext,theta):
    # add an empirical parameter for the phase of the internal field
    gamma = s11(f,f0,g_int,g_ext)
    rotated_gamma = ((gamma-1)*np.exp(1j*theta)) + 1
    x = np.real(rotated_gamma)
    y = np.imag(rotated_gamma)
    return x,y


def s11_empirical_err(params,f,xdata,ydata):
    f0 = params[0]
    g_int = params[1]
    g_ext = params[2]
    theta = params[3]
    x,y = s11_empirical(f,f0,g_int,g_ext,theta)
    err = ((x-xdata)**2)+((y-ydata)**2)
    return err

def get_resonance_params_empirical(freqs, xdata, ydata, \
                         f0_guess, \
                         g_int_guess = 1.0E6, \
                         g_ext_guess = 1.0E6, \
                         theta_guess = 0.0, \
                         return_cost = False, \
                         bounds = True):
    param_guess = [f0_guess, g_int_guess, g_ext_guess, theta_guess]
    if bounds:
        q_bound = freqs[-1]-freqs[0]
        low_bounds = [freqs[0],0,0,-np.pi]
        high_bounds = [freqs[-1],q_bound,q_bound,np.pi]
        bounds_tuple = (low_bounds,high_bounds)
        result = opt.least_squares(s11_empirical_err,param_guess,args=(freqs,xdata,ydata),bounds=bounds_tuple)
    else:
        result = opt.least_squares(s11_empirical_err,param_guess,args=(freqs,xdata,ydata))
    if return_cost:
        return result.x, result.cost
    else:
        return result.x

def s11(f,f0,g_int,g_ext):
    # complex reflection coefficient
    df = f-f0
    numerator = (4*(df**2))+((g_int**2)-(g_ext**2))+(4j*g_ext*df)
    denominator = (4*(df**2))+((g_int + g_ext)**2)
    return numerator/denominator


def s11real(f,f0,g_int,g_ext):
    # x = real, y = imaginary
    gamma = s11(f,f0,g_int,g_ext)
    x = np.real(gamma)
    y = np.imag(gamma)
    return x,y

def s11logmag(f,f0,g_int,g_ext):
    x,y = s11real(f,f0,g_int,g_ext)
    mag = np.sqrt((x**2)+(y**2))
    logmag = 20*(np.log10(mag))
    return logmag


def s11err(params,f,xdata,ydata):
    f0 = params[0]
    g_int = params[1]
    g_ext = params[2]
    x,y = s11real(f,f0,g_int,g_ext)
    err = ((x-xdata)**2)+((y-ydata)**2)
    return err

def get_resonance_params(freqs, xdata, ydata, \
                         f0_guess, \
                         g_int_guess = 5.0E6, \
                         g_ext_guess = 5.0E6, \
                         return_cost = False, \
                         bounds = True):
    param_guess = [f0_guess, g_int_guess, g_ext_guess]
    if bounds:
        q_bound = freqs[-1]-freqs[0]
        low_bounds = [freqs[0],0,0]
        high_bounds = [freqs[-1],q_bound,q_bound]
        bounds_tuple = (low_bounds,high_bounds)
        result = opt.least_squares(s11err,param_guess,args=(freqs,xdata,ydata),bounds=bounds_tuple)
    else:
        result = opt.least_squares(s11err,param_guess,args=(freqs,xdata,ydata))
    if return_cost:
        return result.x, result.cost
    else:
        return result.x
    
def E0_points(n_charges, Vg, Iphi, Ec, Ej, Cg, Vg0, Lphi, Iphi0, ngcrit):
    # Vg, Iphi floats
    # number of charge states = 2*n_charges + 1
    ng = Cg*(Vg-Vg0)
    phi = Lphi*(Iphi-Iphi0)
    
    dim = 2*n_charges+1
    charges = np.linspace(-n_charges,n_charges,dim,dtype=int)
    offdiagvec = np.ones(dim-1)
    diagvec = np.ones(dim)    
    tunnel_coeff = -Ej*np.cos(phi)
    tunnel_mat = tunnel_coeff*(np.diag(offdiagvec,1)+np.diag(offdiagvec,-1))
    charge_coeff = 4*Ec
    for ii in range(dim):
        n = charges[ii]
        if ngcrit < ng%2 < 2-ngcrit:
            diagvec[ii] = (n-((ng+1)/2.0))**2.0
        else:
            diagvec[ii] = (n-(ng/2.0))**2.0
    charge_mat = charge_coeff*np.diag(diagvec)
    mat = charge_mat + tunnel_mat
    val = np.min(np.linalg.eigvalsh(mat))
    return val
    
def E0_arrays(n_charges, Vg, Iphi, Ec, Ej, Cg, Vg0, Lphi, Iphi0, ngcrit):
    # Vg, Iphi arrays
    # number of charge states = 2*n_charges + 1
    ng = Cg*(Vg-Vg0)
    phi = Lphi*(Iphi-Iphi0)
    
    n_gates = len(ng)
    n_fluxes = len(phi)
    vals = np.zeros((n_fluxes,n_gates))
    
    for ii in range(n_fluxes):
        this_phi = phi[ii]
        for jj in range(n_gates):
            this_ng = ng[jj]
            
            dim = 2*n_charges+1
            charges = np.linspace(-n_charges,n_charges,dim,dtype=int)
            offdiagvec = np.ones(dim-1)
            diagvec = np.ones(dim)    
            tunnel_coeff = -Ej*np.cos(this_phi)
            tunnel_mat = tunnel_coeff*(np.diag(offdiagvec,1)+np.diag(offdiagvec,-1))
            charge_coeff = 4*Ec
            for kk in range(dim):
                n = charges[kk]
                if ngcrit < this_ng%2 < 2-ngcrit:
                    diagvec[kk] = (n-((this_ng+1)/2.0))**2.0
                else:
                    diagvec[kk] = (n-(this_ng/2.0))**2.0
            charge_mat = charge_coeff*np.diag(diagvec)
            mat = charge_mat + tunnel_mat
            val = np.min(np.linalg.eigvalsh(mat))
            vals[ii,jj] = val
    return vals
    
def E0_ideal(n_charges, ng, phi, Ec, Ej):
    # ng, phi arrays
    n_gates = len(ng)
    n_fluxes = len(phi)
    vals = np.zeros((n_fluxes,n_gates))
    
    for ii in range(n_fluxes):
        this_phi = phi[ii]
        for jj in range(n_gates):
            this_ng = ng[jj]
            
            dim = 2*n_charges+1
            charges = np.linspace(-n_charges,n_charges,dim,dtype=int)
            offdiagvec = np.ones(dim-1)
            diagvec = np.ones(dim)    
            tunnel_coeff = -Ej*np.cos(this_phi)
            tunnel_mat = tunnel_coeff*(np.diag(offdiagvec,1)+np.diag(offdiagvec,-1))
            charge_coeff = 4*Ec
            for kk in range(dim):
                n = charges[kk]
                diagvec[kk] = (n-(this_ng/2.0))**2.0
            charge_mat = charge_coeff*np.diag(diagvec)
            mat = charge_mat + tunnel_mat
            val = np.min(np.linalg.eigvalsh(mat))
            vals[ii,jj] = val
    return vals

def omegaR(n_charges, Vg, Iphi, omega0, Ec, Ej, Cg, Vg0, Lphi, Iphi0, ngcrit):
    phizp = 0.0878776
    phi = Lphi*(Iphi-Iphi0)
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    E0 = E0_arrays(n_charges, Vg, Iphi, Ec, Ej, Cg, Vg0, Lphi, Iphi0, ngcrit)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi**2)
    omega = omega0 + 2*(phizp**2)*d2E0
    return omega
    
def omegaR_ideal(n_charges, ng, phi, Ec, Ej):
    phizp = 0.0878776
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    E0 = E0_ideal(n_charges, ng, phi, Ec, Ej)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi**2)
    omegaR = (phizp**2)*d2E0
    return omegaR[1:-1]

def fR_fitting_miles(params, inputs):
    # input fc, fj in Hz
    f0, fc, fj, Cg, Vg0, Lphi, Iphi0, ngcrit = params
    Iphi, Vg, n_charges = inputs
    phizp = 0.0878776
    phi = Lphi*(Iphi-Iphi0)
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    E0 = E0_arrays(n_charges, Vg, Iphi, fc, fj, Cg, Vg0, Lphi, Iphi0, ngcrit)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi**2)
    f = f0 + (phizp**2)*d2E0
    return f[1:-1]

def get_band_renormalization_params(Iphi, Vg, n_charges, f0_data, f0, fc, fj, Cg, Vg0, Lphi, Iphi0, ngcrit, return_covar = True):
    param_guesses = f0, fc, fj, Cg, Vg0, Lphi, Iphi0, ngcrit
    params, covar = opt.curve_fit(fR_fitting_miles,(Iphi,Vg,n_charges),f0_data,param_guesses)
    if return_covar:
        return params,covar
    else:
        return params
    
def fR_miles_err(params, Iphi, Vg, n_charges, f0_data):
    inputs = Iphi, Vg, n_charges
    fR = fR_fitting_miles(params, inputs)
    err = np.sum(fR - f0_data[1:-1])
    return err

def get_band_renormalization_params_lsq(Iphi, Vg, n_charges, f0_data, f0, fc, fj, Cg, Vg0, Lphi, Iphi0, ngcrit, return_cost = True):
    param_guesses = [f0, fc, fj, Cg, Vg0, Lphi, Iphi0, ngcrit]
    result = opt.least_squares(fR_miles_err,param_guesses,args=(Iphi,Vg,n_charges,f0_data))
    if return_cost:
        return result.x, result.cost
    else:
        return result.x
    
def fR_fitting(n_charges, Vg, Iphi, omega0, Ec, Ej, Cg, Vg0, Lphi, Iphi0, ngcrit):
    pass

def fR_miles_steepest_descent(params,param_steps,Iphi,Vg,n_charges,f0_data):
    inputs = Iphi, Vg, n_charges
    fR = fR_fitting_miles(params, inputs)
    dummy_params = tuple(params)
    for ii in range(len(params)):
        new_param = params[ii]+param_steps[ii]
    pass
    
    
def fR_simple(n_charges, Vg, Iphi, fc, fj):
    Vg0 = 0.01054054
    Cg = 40.65
    Iphi0 = -1.024E-5
    Lphi = 69804
    
        





def s11(f,f0,g_int,g_ext):
    # complex reflection coefficient
    df = f-f0
    numerator = (4*(df**2))+((g_int**2)-(g_ext**2))+(4j*g_ext*df)
    denominator = (4*(df**2))+((g_int + g_ext)**2)
    return numerator/denominator

def s11_empirical(f,f0,g_int,g_ext,theta):
    # add an empirical parameter for the phase of the internal field
    df = f-f0
    numerator = (4*(df**2))+((g_int**2)-(g_ext**2))+(4j*g_ext*df)
    denominator = (4*(df**2))+((g_int + g_ext)**2)
    gamma = numerator/denominator
    rotated_gamma = ((gamma-1)*np.exp(1j*theta)) + 1
    x = np.real(rotated_gamma)
    y = np.imag(rotated_gamma)
    return x,y


def s11_emp_fitting(f,f0,g_int,g_ext,theta):
    # add an empirical parameter for the phase of the internal field
    df = f-f0
    numerator = (4*(df**2))+((g_int**2)-(g_ext**2))+(4j*g_ext*df)
    denominator = (4*(df**2))+((g_int + g_ext)**2)
    gamma = numerator/denominator
    rotated_gamma = ((gamma-1)*np.exp(1j*theta)) + 1
    result = np.hstack((np.real(rotated_gamma),np.imag(rotated_gamma)))
    return result

def s11_noisy_fitting_analytic(f,f0,g_int,g_ext,sigma,theta):
    df = f-f0
    term = (df-(1j*((g_int+g_ext)/2)))/(sigma*np.sqrt(2))
    gamma = 1+(((1j*np.pi*g_ext)/(np.sqrt(2*np.pi*(sigma**2))))*np.exp(-1*(term**2))*(1j+sp.special.erfi(term)))
    rotated_gamma = ((gamma-1)*np.exp(1j*theta)) + 1
    result = np.hstack((np.real(rotated_gamma),np.imag(rotated_gamma)))
    return result

def s11_noisy_fitting_numeric(f,f0,g_int,g_ext,sigma,theta):
    n_samples = 1000
    rands = np.random.normal(0,sigma,size=n_samples)
    avg_gamma = np.zeros(2*len(f),dtype=float)
    for ii in range(n_samples):
        df = f-f0-rands[ii]
        numerator = (1j*df)+((g_int-g_ext)/2)
        denominator = (1j*df)+((g_int+g_ext)/2)
        gamma = numerator/denominator
        rotated_gamma = ((gamma-1)*np.exp(1j*theta)) + 1
        avg_gamma += np.hstack((np.real(rotated_gamma),np.imag(rotated_gamma)))
    avg_gamma = avg_gamma/n_samples
    return avg_gamma

def fit_reflection(freqs,data, \
                        f0_guess = None, \
                        g_int_guess = 1.0E6, \
                        g_ext_guess = 1.0E6, \
                        theta_guess = 0.0, \
                        bounds = True):
    param_guess = [f0_guess, g_int_guess, g_ext_guess, theta_guess]
    if not f0_guess:
        f0_guess = freqs[round(len(freqs)/2)]
    if bounds:
        q_bound = freqs[-1]-freqs[0]
        low_bounds = [freqs[0],0,0,-np.pi]
        high_bounds = [freqs[-1],q_bound,q_bound,np.pi]
        bounds_tuple = (low_bounds,high_bounds)
    else:
        low_bounds = [-np.inf,-np.inf,-np.inf,-np.inf]
        high_bounds = [np.inf,np.inf,np.inf,np.inf]
        bounds_tuple = (low_bounds,high_bounds)
    popt,pcov = opt.curve_fit(s11_emp_fitting, \
                              freqs, \
                              data, \
                              p0=param_guess)
    return popt,pcov



def fit_noisy_reflection_analytic(freqs,data, \
                        f0_guess = None, \
                        g_int_guess = 0.3E6, \
                        g_ext_guess = 1.3E6, \
                        sigma_guess = 0.5E6, \
                        theta_guess = 0.0, \
                        bounds = True):
    if not f0_guess:
        f0_guess = freqs[round(len(freqs)/2)]
    param_guess = [f0_guess, g_int_guess, g_ext_guess, sigma_guess, theta_guess]
    if bounds:
        q_bound = freqs[-1]-freqs[0]
        low_bounds = [freqs[0],0.1E6,0,0,-np.pi]
        high_bounds = [freqs[-1],q_bound,q_bound,q_bound,np.pi]
        bounds_tuple = (low_bounds,high_bounds)
    else:
        low_bounds = [-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]
        high_bounds = [np.inf,np.inf,np.inf,np.inf,np.inf]
        bounds_tuple = (low_bounds,high_bounds)
    popt,pcov = opt.curve_fit(s11_noisy_fitting_analytic, \
                              freqs, \
                              data, \
                              p0=param_guess, \
                              bounds=bounds_tuple)
    perr = np.sqrt(np.diag(pcov))
    return popt,perr
        
def fit_noisy_reflection_numeric(freqs,data, \
                        f0_guess = None, \
                        g_int_guess = 0.5E6, \
                        g_ext_guess = 1.5E6, \
                        sigma_guess = 0.5E6, \
                        theta_guess = 0.0, \
                        bounds = True):
    param_guess = [f0_guess, g_int_guess, g_ext_guess, sigma_guess, theta_guess]
    if not f0_guess:
        f0_guess = freqs[round(len(freqs)/2)]
    if bounds:
        q_bound = freqs[-1]-freqs[0]
        low_bounds = [freqs[0],0,0,0,-np.pi]
        high_bounds = [freqs[-1],q_bound,q_bound,q_bound,np.pi]
        bounds_tuple = (low_bounds,high_bounds)
    else:
        low_bounds = [-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]
        high_bounds = [np.inf,np.inf,np.inf,np.inf,np.inf]
        bounds_tuple = (low_bounds,high_bounds)
    popt,pcov = opt.curve_fit(s11_noisy_fitting_numeric, \
                              freqs, \
                              data, \
                              p0=param_guess, \
                              bounds=bounds_tuple)
    return popt,pcov
        


def plot_reflection_coefficient_fits_empirical(freqs, xdata, ydata, f0, g_int, g_ext, theta_int, scale=5):
    gamma_data = xdata+(1j*ydata)
    gamma_data = ((gamma_data-1.0)*np.exp(-1j*theta_int))+1.0
    xdata = np.real(gamma_data)
    ydata = np.imag(gamma_data)
    
    linmag = np.sqrt((xdata**2)+(ydata**2))
    phase = (180.0/np.pi)*np.arctan2(ydata,xdata)
    
    xfit, yfit = s11real(freqs, f0, g_int, g_ext)
    linmag_fit = np.sqrt((xfit**2)+(yfit**2))
    phase_fit = (180.0/np.pi)*np.arctan2(yfit, xfit)
        
    f,axarr = plt.subplots(1,3,figsize=(3*scale, scale))
    axarr[0].plot(xdata,ydata,linestyle='None',marker='.')
    axarr[0].plot(xfit,yfit,linewidth=2.0)
    axarr[0].set(xlabel='Re[S11]',ylabel='Im[S11]')
    #axarr[0].set_aspect(2.0)
    axarr[1].plot(freqs,linmag,linestyle='None',marker='.')
    axarr[1].plot(freqs,linmag_fit,linewidth=2.0)
    axarr[1].set(xlabel='Frequency (Hz)',ylabel='|S11|')
    #axarr[1].set_aspect(1.0)
    axarr[2].plot(freqs,phase,linestyle='None',marker='.')
    axarr[2].plot(freqs,phase_fit,linewidth=2.0)
    axarr[2].set(xlabel='Frequency (Hz)',ylabel='arg[S11] (degrees)')
    plt.tight_layout()
    return f

def plot_reflection_coefficient_fits_rotated(freqs, xdata, ydata, gamma_fit, theta_int, scale=5):
    gamma_data = xdata+(1j*ydata)
    gamma_data = ((gamma_data-1.0)*np.exp(-1j*theta_int))+1.0
    xdata = np.real(gamma_data)
    ydata = np.imag(gamma_data)
    linmag = np.sqrt((xdata**2)+(ydata**2))
    phase = (180.0/np.pi)*np.arctan2(ydata,xdata)
    
    gamma_fit = ((gamma_fit-1.0)*np.exp(-1j*theta_int))+1.0
    xfit = np.real(gamma_fit)
    yfit = np.imag(gamma_fit)
    linmag_fit = np.sqrt((xfit**2)+(yfit**2))
    phase_fit = (180.0/np.pi)*np.arctan2(yfit, xfit)
        
    f,axarr = plt.subplots(1,3,figsize=(3*scale, scale))
    axarr[0].plot(xdata,ydata,linestyle='None',marker='.')
    axarr[0].plot(xfit,yfit,linewidth=2.0)
    axarr[0].set(xlabel='Re[S11]',ylabel='Im[S11]')
    #axarr[0].set_aspect(2.0)
    axarr[1].plot(freqs,linmag,linestyle='None',marker='.')
    axarr[1].plot(freqs,linmag_fit,linewidth=2.0)
    axarr[1].set(xlabel='Frequency (Hz)',ylabel='|S11|')
    #axarr[1].set_aspect(1.0)
    axarr[2].plot(freqs,phase,linestyle='None',marker='.')
    axarr[2].plot(freqs,phase_fit,linewidth=2.0)
    axarr[2].set(xlabel='Frequency (Hz)',ylabel='arg[S11] (degrees)')
    plt.tight_layout()
    return f


def visualize_reflection_coefficient(freqs,xdata,ydata,scale=4):  
    linmag = np.sqrt((xdata**2)+(ydata**2))
    phase = (180.0/np.pi)*np.arctan2(ydata,xdata)
        
    f,axarr = plt.subplots(1,3,figsize=(3*scale, scale))
    axarr[0].plot(xdata,ydata,linestyle='None',marker='.')
    axarr[0].set(xlabel='Re[S11]',ylabel='Im[S11]')
    #axarr[0].set_aspect(2.0)
    axarr[1].plot(freqs,linmag,linestyle='None',marker='.')
    axarr[1].set(xlabel='Frequency (Hz)',ylabel='|S11|')
    #axarr[1].set_aspect(1.0)
    axarr[2].plot(freqs,phase,linestyle='None',marker='.')
    axarr[2].set(xlabel='Frequency (Hz)',ylabel='arg[S11] (degrees)')
    plt.tight_layout()
    return f

def E0_CPT(phis, ngs, fc, fj, n_charges=2):
    # number of charge states = 2*n_charges + 1   
    n_gates = len(ngs)
    n_fluxes = len(phis)
    vals = np.zeros((n_fluxes,n_gates))
    for ii in range(n_fluxes):
        this_phi = phis[ii]
        for jj in range(n_gates):
            this_ng = ngs[jj]
            dim = 2*n_charges+1
            charges = np.linspace(-n_charges,n_charges,dim,dtype=int)
            offdiagvec = np.ones(dim-1)
            diagvec = np.ones(dim)
            tunnel_coeff = -fj*np.cos(this_phi)
            tunnel_mat = tunnel_coeff*(np.diag(offdiagvec,1)+np.diag(offdiagvec,-1))
            charge_coeff = 4*fc
            for kk in range(dim):
                n = charges[kk]
                diagvec[kk] = (n-(this_ng/2.0))**2.0
            charge_mat = charge_coeff*np.diag(diagvec)
            mat = charge_mat + tunnel_mat
            val = np.min(np.linalg.eigvalsh(mat))
            vals[ii,jj] = val
    return vals

def E0_arrays(n_charges, Vg, Iphi, Ec, Ej, Cg, Vg0, Lphi, Iphi0):
    # Vg, Iphi arrays
    # number of charge states = 2*n_charges + 1
    ng = Cg*(Vg-Vg0)
    phi = Lphi*(Iphi-Iphi0)
    
    n_gates = len(ng)
    n_fluxes = len(phi)
    vals = np.zeros((n_fluxes,n_gates))
    
    for ii in range(n_fluxes):
        this_phi = phi[ii]
        for jj in range(n_gates):
            this_ng = ng[jj]
            
            dim = 2*n_charges+1
            charges = np.linspace(-n_charges,n_charges,dim,dtype=int)
            offdiagvec = np.ones(dim-1)
            diagvec = np.ones(dim)    
            tunnel_coeff = -Ej*np.cos(this_phi)
            tunnel_mat = tunnel_coeff*(np.diag(offdiagvec,1)+np.diag(offdiagvec,-1))
            charge_coeff = 4*Ec
            for kk in range(dim):
                n = charges[kk]
                diagvec[kk] = (n-(this_ng/2.0))**2.0
            charge_mat = charge_coeff*np.diag(diagvec)
            mat = charge_mat + tunnel_mat
            val = np.min(np.linalg.eigvalsh(mat))
            vals[ii,jj] = val
    return vals

def fR(Vg, Iphi, fc, fj, f0, Cg, Vg0, Lphi, Iphi0, n_charges):
    phizp = 0.0878776
    phi = Lphi*(Iphi-Iphi0)
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    E0 = E0_arrays(n_charges, Vg, Iphi, fc, fj, Cg, Vg0, Lphi, Iphi0)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi**2)
    f = f0 + (phizp**2)*d2E0
    return f

def fR_simple(fc, fj, Iphi, Vg, n_charges):
    Vg0 = 0.01054054
    Cg = 40.65
    Iphi0 = -1.025E-5
    Lphi = 69804
    f0 = 5.75733526E9
    return fR(Vg,Iphi,fc, fj, f0, Cg, Vg0, Lphi, Iphi0, n_charges)

def E0_ideal(n_charges, ng, phi, Ec, Ej):
    # ng, phi arrays
    n_gates = len(ng)
    n_fluxes = len(phi)
    vals = np.zeros((n_fluxes,n_gates))
    
    for ii in range(n_fluxes):
        this_phi = phi[ii]
        for jj in range(n_gates):
            this_ng = ng[jj]
            
            dim = 2*n_charges+1
            charges = np.linspace(-n_charges,n_charges,dim,dtype=int)
            offdiagvec = np.ones(dim-1)
            diagvec = np.ones(dim)    
            tunnel_coeff = -Ej*np.cos(this_phi)
            tunnel_mat = tunnel_coeff*(np.diag(offdiagvec,1)+np.diag(offdiagvec,-1))
            charge_coeff = 4*Ec
            for kk in range(dim):
                n = charges[kk]
                diagvec[kk] = (n-(this_ng/2.0))**2.0
            charge_mat = charge_coeff*np.diag(diagvec)
            mat = charge_mat + tunnel_mat
            val = np.min(np.linalg.eigvalsh(mat))
            vals[ii,jj] = val
    return vals

def fR_ideal(n_charges, ng, phi, fc, fj, f0):
    phizp = 0.0878776
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    E0 = E0_ideal(n_charges, ng, phi, fc, fj)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi**2)
    fR = (phizp**2)*d2E0
    return f0 + fR[1:-1]

def fR_fitting(params,inputs):
    Iphi, Vg, n_charges = inputs
    fc, fj = params
    fR = fR_simple(fc, fj, Iphi, Vg, n_charges)
    return fR[1:-1]
    
def fR_err(params, Iphi, Vg, n_charges, f0_data):
    inputs = Iphi, Vg, n_charges
    fR = fR_fitting(params, inputs)
    err = np.sum((fR - f0_data[1:-1])**2)
    return err

    
def get_band_renormalization_params(Iphi, Vg, n_charges, f0_data, fc, fj, return_cost = True):
    param_guesses = [fc, fj]
    result = opt.least_squares(fR_err,param_guesses,args=(Iphi,Vg,n_charges,f0_data))
    if return_cost:
        return result.x, result.cost
    else:
        return result.x

def fR_interp(Vg, Iphi, fc, fj, f0, Cg, Vg0, Lphi, Iphi0, n_charges, interp):
    Vg0 = 0.01204054
    Cg = 40.65
    Iphi0 = -1.025E-5
    Lphi = 69804
    f0 = 5.75733526E9
    phizp = 0.0878776
    phi = Lphi*(Iphi-Iphi0)
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    E0 = E0_arrays(n_charges, Vg, Iphi, fc, fj, Cg, Vg0, Lphi, Iphi0)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi**2)
    f = f0 + (phizp**2)*d2E0
    return f

def fR_simple_interp(fc, fj, Iphi, Vg, n_charges, interp):    
    Vg0 = 0.01204054
    Cg = 40.65
    Iphi0 = -1.025E-5
    Lphi = 69804
    f0 = 5.75733526E9
    phizp = 0.0878776
    
    phi = Lphi*(Iphi-Iphi0)
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    
    Iphi_interp = np.linspace(Iphi[0],Iphi[-1],interp*len(Iphi))
    phi_interp = Lphi*(Iphi_interp-Iphi0)
    deltaphi_interp = phi_interp-np.roll(phi_interp,1)
    dphi_interp = np.mean(deltaphi_interp[1:])
    
    E0 = E0_arrays(n_charges, Vg, Iphi_interp, fc, fj, Cg, Vg0, Lphi, Iphi0)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi_interp**2)
    f_interp = f0 + (phizp**2)*d2E0
    f_interp = f_interp[1:-1]
    
    f = np.zeros((len(Iphi[1:-1]),len(Vg)))
    for ii in range(len(Iphi[1:-1])):
        for jj in range(len(Vg)):
            f[ii,jj] = f_interp[interp*ii,jj]
            
    return f

def fR_fitting_interp(params,inputs):
    Iphi, Vg, n_charges, interp = inputs
    fc, fj = params
    fR = fR_simple_interp(fc, fj, Iphi, Vg, n_charges, interp)
    return fR[1:-1]
    
def fR_err_interp(params, Iphi, Vg, n_charges, f0_data, interp):
    fc, fj = params
    
    Vg0 = 0.01204054
    Cg = 40.65
    Iphi0 = -1.025E-5
    Lphi = 69804
    f0 = 5.75733526E9
    phizp = 0.0878776
    
    phi = Lphi*(Iphi-Iphi0)
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    
    Iphi_interp = np.linspace(Iphi[0],Iphi[-1],interp*len(Iphi))
    phi_interp = Lphi*(Iphi_interp-Iphi0)
    deltaphi_interp = phi_interp-np.roll(phi_interp,1)
    dphi_interp = np.mean(deltaphi_interp[1:])
    
    E0 = E0_arrays(n_charges, Vg, Iphi_interp, fc, fj, Cg, Vg0, Lphi, Iphi0)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi_interp**2)
    f_interp = f0 + (phizp**2)*d2E0
    f_interp = f_interp[1:-1]
    
    f = np.zeros((len(Iphi[1:-1]),len(Vg)))
    for ii in range(len(Iphi[1:-1])):
        for jj in range(len(Vg)):
            f[ii,jj] = f_interp[interp*ii,jj]
    
    err = np.sum((f - f0_data[1:-1])**2)
    return err
    
def get_band_renormalization_params_interp(Iphi, Vg, n_charges, f0_data, fc, fj, interp = 3, return_cost = True):
    param_guesses = [fc, fj]
    result = opt.least_squares(fR_err_interp,param_guesses,args=(Iphi,Vg,n_charges,f0_data,interp))
    if return_cost:
        return result.x, result.cost
    else:
        return result.x


def fR(Vg, Iphi, fc, fj, f0, Cg, Vg0, Lphi, Iphi0, n_charges):
    phizp = 0.0878776
    phi = Lphi*(Iphi-Iphi0)
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    E0 = E0_arrays(n_charges, Vg, Iphi, fc, fj, Cg, Vg0, Lphi, Iphi0)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi**2)
    f = f0 + (phizp**2)*d2E0
    return f

def fR_simple(fc, fj, Iphi, Vg, n_charges):
    Vg0 = 0.01054054
    Cg = 40.65
    Iphi0 = -1.025E-5
    Lphi = 69804
    f0 = 5.75733526E9
    return fR(Vg,Iphi,fc, fj, f0, Cg, Vg0, Lphi, Iphi0, n_charges)

def E0_ideal(n_charges, ng, phi, Ec, Ej):
    # ng, phi arrays
    n_gates = len(ng)
    n_fluxes = len(phi)
    vals = np.zeros((n_fluxes,n_gates))
    
    for ii in range(n_fluxes):
        this_phi = phi[ii]
        for jj in range(n_gates):
            this_ng = ng[jj]
            
            dim = 2*n_charges+1
            charges = np.linspace(-n_charges,n_charges,dim,dtype=int)
            offdiagvec = np.ones(dim-1)
            diagvec = np.ones(dim)    
            tunnel_coeff = -Ej*np.cos(this_phi)
            tunnel_mat = tunnel_coeff*(np.diag(offdiagvec,1)+np.diag(offdiagvec,-1))
            charge_coeff = 4*Ec
            for kk in range(dim):
                n = charges[kk]
                diagvec[kk] = (n-(this_ng/2.0))**2.0
            charge_mat = charge_coeff*np.diag(diagvec)
            mat = charge_mat + tunnel_mat
            val = np.min(np.linalg.eigvalsh(mat))
            vals[ii,jj] = val
    return vals

def fR_ideal(n_charges, ng, phi, fc, fj, f0):
    phizp = 0.0878776
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    E0 = E0_ideal(n_charges, ng, phi, fc, fj)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi**2)
    fR = (phizp**2)*d2E0
    return f0 + fR[1:-1]

def fR_fitting(params,inputs):
    Iphi, Vg, n_charges = inputs
    fc, fj = params
    fR = fR_simple(fc, fj, Iphi, Vg, n_charges)
    return fR[1:-1]
    
def fR_err(params, Iphi, Vg, n_charges, f0_data):
    inputs = Iphi, Vg, n_charges
    fR = fR_fitting(params, inputs)
    err = np.sum((fR - f0_data[1:-1])**2)
    return err

    
def get_band_renormalization_params(Iphi, Vg, n_charges, f0_data, fc, fj, return_cost = True):
    param_guesses = [fc, fj]
    result = opt.least_squares(fR_err,param_guesses,args=(Iphi,Vg,n_charges,f0_data))
    if return_cost:
        return result.x, result.cost
    else:
        return result.x

def fR_interp(Vg, Iphi, fc, fj, f0, Cg, Vg0, Lphi, Iphi0, n_charges, interp):
    Vg0 = 0.01204054
    Cg = 40.65
    Iphi0 = -1.025E-5
    Lphi = 69804
    f0 = 5.75733526E9
    phizp = 0.0878776
    phi = Lphi*(Iphi-Iphi0)
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    E0 = E0_arrays(n_charges, Vg, Iphi, fc, fj, Cg, Vg0, Lphi, Iphi0)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi**2)
    f = f0 + (phizp**2)*d2E0
    return f

def fR_simple_interp(fc, fj, Iphi, Vg, n_charges, interp):    
    Vg0 = 0.01204054
    Cg = 40.65
    Iphi0 = -1.025E-5
    Lphi = 69804
    f0 = 5.75733526E9
    phizp = 0.0878776
    
    phi = Lphi*(Iphi-Iphi0)
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    
    Iphi_interp = np.linspace(Iphi[0],Iphi[-1],interp*len(Iphi))
    phi_interp = Lphi*(Iphi_interp-Iphi0)
    deltaphi_interp = phi_interp-np.roll(phi_interp,1)
    dphi_interp = np.mean(deltaphi_interp[1:])
    
    E0 = E0_arrays(n_charges, Vg, Iphi_interp, fc, fj, Cg, Vg0, Lphi, Iphi0)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi_interp**2)
    f_interp = f0 + (phizp**2)*d2E0
    f_interp = f_interp[1:-1]
    
    f = np.zeros((len(Iphi[1:-1]),len(Vg)))
    for ii in range(len(Iphi[1:-1])):
        for jj in range(len(Vg)):
            f[ii,jj] = f_interp[interp*ii,jj]
            
    return f

def fR_fitting_interp(params,inputs):
    Iphi, Vg, n_charges, interp = inputs
    fc, fj = params
    fR = fR_simple_interp(fc, fj, Iphi, Vg, n_charges, interp)
    return fR[1:-1]
    
def fR_err_interp(params, Iphi, Vg, n_charges, f0_data, interp):
    fc, fj = params
    
    Vg0 = 0.01204054
    Cg = 40.65
    Iphi0 = -1.025E-5
    Lphi = 69804
    f0 = 5.75733526E9
    phizp = 0.0878776
    
    phi = Lphi*(Iphi-Iphi0)
    deltaphi = phi-np.roll(phi,1)
    dphi = np.mean(deltaphi[1:])
    
    Iphi_interp = np.linspace(Iphi[0],Iphi[-1],interp*len(Iphi))
    phi_interp = Lphi*(Iphi_interp-Iphi0)
    deltaphi_interp = phi_interp-np.roll(phi_interp,1)
    dphi_interp = np.mean(deltaphi_interp[1:])
    
    E0 = E0_arrays(n_charges, Vg, Iphi_interp, fc, fj, Cg, Vg0, Lphi, Iphi0)
    E0prev = np.roll(E0,1,axis=0)
    E0next = np.roll(E0,-1,axis=0)
    d2E0 = (E0prev - 2*E0 + E0next)/(dphi_interp**2)
    f_interp = f0 + (phizp**2)*d2E0
    f_interp = f_interp[1:-1]
    
    f = np.zeros((len(Iphi[1:-1]),len(Vg)))
    for ii in range(len(Iphi[1:-1])):
        for jj in range(len(Vg)):
            f[ii,jj] = f_interp[interp*ii,jj]
    
    err = np.sum((f - f0_data[1:-1])**2)
    return err
    
def get_band_renormalization_params_interp(Iphi, Vg, n_charges, f0_data, fc, fj, interp = 3, return_cost = True):
    param_guesses = [fc, fj]
    result = opt.least_squares(fR_err_interp,param_guesses,args=(Iphi,Vg,n_charges,f0_data,interp))
    if return_cost:
        return result.x, result.cost
    else:
        return result.x
    