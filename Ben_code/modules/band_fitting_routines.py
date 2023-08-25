#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 18:42:10 2019

@author: ben
"""
import scipy as scp
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


def make_f0ccpt(n_charges, f0_bare = 5.7574E9, fit_duffing = False):
    # make f0_bare=None to fit to it
    if isinstance(f0_bare,type(None)):
        fit_f0_bare = True
    else:
        fit_f0_bare = False
        
    if fit_duffing:
        if fit_f0_bare:
            def f0ccpt(biases,fc,fj,f0_bare):
                phizp = 0.0878776
                phis = biases[0]
                ngs = biases[1]
                d2f = dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
                d4f = dfCPT(ngs,phis,fc,fj,4,n_charges=n_charges)
                f0 = f0_bare + ((phizp**2)*d2f) + ((0.25*(phizp**4))*d4f)
                return f0.ravel()
        else:
            def f0ccpt(biases,fc,fj):
                #f0_bare = 5.75733526E9
                #f0_bare = 5.7577E9
                phizp = 0.0878776
                phis = biases[0]
                ngs = biases[1]
                d2f = dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
                d4f = dfCPT(ngs,phis,fc,fj,4,n_charges=n_charges)
                f0 = f0_bare + ((phizp**2)*d2f) + ((0.25*(phizp**4))*d4f)
                return f0.ravel()
    else:
        if fit_f0_bare:
            def f0ccpt(biases,fc,fj,f0_bare):
                phizp = 0.0878776
                phis = biases[0]
                ngs = biases[1]
                d2f = dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
                f0 = f0_bare + ((phizp**2)*d2f)
                return f0.ravel()
        else:
            def f0ccpt(biases,fc,fj):
                #f0_bare = 5.75733526E9
                #f0_bare = 5.7577E9
                phizp = 0.0878776
                phis = biases[0]
                ngs = biases[1]
                d2f = dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
                f0 = f0_bare + ((phizp**2)*d2f)
                return f0.ravel()
    return f0ccpt
        
def fit_f0ccpt(phis, ngs, f0_array, \
               fc_guess = 50.0E9, \
               fj_guess = 15.0E9, \
               n_charges = 2, \
               f0_bare = 5.7574E9, \
               f0_bare_guess = 5.7574E9, \
               fit_duffing = False, \
               bounds = True, \
               tolerance = 1.0E-15, \
               fscale = 1.0E6, \
               n_evals = 1E7):
    biases = [phis,ngs]
    f0_data = f0_array.ravel()
    param_guess = [fc_guess,fj_guess,f0_bare_guess]
    
    if isinstance(f0_bare,type(None)):
        n_params = 3
    else:
        n_params = 2
    
    param_guess = param_guess[0:n_params]
    if bounds:
        low_bounds = [1.0E9,1.0E9,5.7E9]
        low_bounds = low_bounds[0:n_params]
        high_bounds = [100.0E9,100.0E9,5.8E9]
        high_bounds = high_bounds[0:n_params]
    else:
        low_bounds = [0.0,0.0,0.0]
        low_bounds = low_bounds[0:n_params]
        high_bounds = [np.inf,np.inf,np.inf]
        high_bounds = high_bounds[0:n_params]
    bounds_tuple = (low_bounds,high_bounds)
    
    fscales = [fscale,fscale,fscale]
    fscales = fscales[0:n_params]
    
    popt,pcov = opt.curve_fit(make_f0ccpt(n_charges,f0_bare,fit_duffing), \
                              biases, \
                              f0_data, \
                              p0=param_guess, \
                              bounds=bounds_tuple, \
                              ftol=tolerance, \
                              xtol=tolerance, \
                              gtol=tolerance, \
                              x_scale = fscales, \
                              max_nfev=n_evals)
    perr = np.sqrt(np.diag(pcov))
    return popt,perr

def get_f0ccpt_fit(phis,ngs,fc,fj, \
                   n_charges=2, \
                   f0_bare = 5.75733526E9, \
                   fit_duffing = False):
    # make f0_bare=None to fit to it
    phizp = 0.0878776
    if isinstance(f0_bare,type(None)):
        fit_f0_bare = True
    else:
        fit_f0_bare = False
    if fit_duffing:
        if fit_f0_bare:
            d2f = dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
            d4f = dfCPT(ngs,phis,fc,fj,4,n_charges=n_charges)
            f0 = f0_bare + ((phizp**2)*d2f) + ((0.25*(phizp**4))*d4f)
        else:
            #f0_bare = 5.75733526E9
            #f0_bare = 5.7577E9
            d2f = dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
            d4f = dfCPT(ngs,phis,fc,fj,4,n_charges=n_charges)
            f0 = f0_bare + ((phizp**2)*d2f) + ((0.25*(phizp**4))*d4f)
    else:
        if fit_f0_bare:
            d2f = dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
            f0 = f0_bare + ((phizp**2)*d2f)
        else:
            #f0_bare = 5.75733526E9
            #f0_bare = 5.7577E9
            d2f = dfCPT(ngs,phis,fc,fj,2,n_charges=n_charges)
            f0 = f0_bare + ((phizp**2)*d2f)
    return f0

def fCPT(ngs, phi_exts, fc, fj, n_charges = 2):
    # number of charge states = 2*n_charges + 1
    single_gate = isinstance(ngs,type(1.3))
    single_flux = isinstance(phi_exts,type(1.3))
    if single_gate:
        n_gates = 1
    else:
        n_gates = len(ngs)
    if single_flux:
        n_fluxes = 1
    else:
        n_fluxes = len(phi_exts)
    vals = np.zeros((n_fluxes,n_gates))
    for ii in range(n_fluxes):
        if single_flux:
            this_phi = phi_exts
        else:
            this_phi = phi_exts[ii]
        for jj in range(n_gates):
            if single_gate:
                this_ng = ngs
            else:
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

def avg_cos_dphi(ngs, phi_exts, fc, fj, n_charges=2):
    # number of charge states = 2*n_charges + 1
    n_gates = len(ngs)
    n_fluxes = len(phi_exts)
    cos_dphis = np.zeros((n_fluxes,n_gates))
    for ii in range(n_fluxes):
        this_phi = phi_exts[ii]
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
            vals,vecs = np.linalg.eigh(mat)
            vec = vecs[:,np.argmin(vals)]
            cos_dphi_mat = tunnel_mat/(2.0*tunnel_coeff)
            cos_dphis[ii,jj] = np.dot(vec.T.conjugate(),np.dot(cos_dphi_mat,vec))
    return cos_dphis

def var_cos_dphi(ngs, phi_exts, fc, fj, n_charges=2):
    # number of charge states = 2*n_charges + 1
    n_gates = len(ngs)
    n_fluxes = len(phi_exts)
    var_cos_dphis = np.zeros((n_fluxes,n_gates))
    for ii in range(n_fluxes):
        this_phi = phi_exts[ii]
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
            vals,vecs = np.linalg.eigh(mat)
            vec = vecs[:,np.argmin(vals)]
            cos_dphi_mat = tunnel_mat/(2.0*tunnel_coeff)
            cos_dphi_squared_mat = np.dot(cos_dphi_mat,cos_dphi_mat)
            var_cos_dphis[ii,jj] = np.dot(vec.T.conjugate(),np.dot(cos_dphi_squared_mat,vec))
    return var_cos_dphis

def avg_sin_dphi(ngs, phi_exts, fc, fj, n_charges=2):
    # number of charge states = 2*n_charges + 1
    if isinstance(ngs,type(1.3)):
        n_gates = 1
    else:
        n_gates = len(ngs)
    if isinstance(phi_exts,type(1.3)):
        n_fluxes = 1
    else:
        n_fluxes = len(phi_exts)
    sin_dphis = np.zeros((n_fluxes,n_gates))
    for ii in range(n_fluxes):
        this_phi = phi_exts[ii]
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
            vals,vecs = np.linalg.eigh(mat)
            vec = vecs[:,np.argmin(vals)]
#            print('vals:')
#            print(vals)
            sin_dphi_mat = 1j*0.5*(np.diag(offdiagvec,1)-np.diag(offdiagvec,-1))
            this_sin_dphi = np.dot(vec.T.conjugate(),np.dot(sin_dphi_mat,vec))
#            print('sin_dphi:')
#           print(this_sin_dphi)
            sin_dphis[ii,jj] = np.abs(this_sin_dphi)
#            print('vec:')
#            print(vec)
#            print('adjoint vec:')
#            print(vec.T.conjugate())
#            print('sindphi:')
#            print(sin_dphis[ii,jj])
#            stop_flag = input('stop?')
#            if stop_flag:
#                sys.exit()
    return sin_dphis
    

def dfCPT(ngs, phi_exts, fc, fj, order, n_charges = 2):
    extra_pts = math.ceil(order/2)
    stride = (2*extra_pts)+1
    if isinstance(phi_exts,type(5.0)):
        dphi = 0.001
        phi_interp = np.empty(stride,dtype=float)
    else:
        dphi = np.min(phi_exts[1:]-np.roll(phi_exts,1)[1:])/(10*extra_pts)
        phi_interp = np.empty(stride*len(phi_exts),dtype=float)
    for ii in range(stride):
        phi_interp[ii::stride] = phi_exts + ((ii-extra_pts)*dphi)
    f = fCPT(ngs, phi_interp, fc, fj, n_charges)
    if order == 1:
        coeffs = [-1/2, 0, 1/2]
    elif order == 2:
        coeffs = [1, -2, 1]
    elif order == 3:
        coeffs = [-1/2, 1, 0, -1, 1/2]
    elif order == 4:
        coeffs = [1, -4, 6, -4, 1]
    elif order == 5:
        coeffs = [-1/2, 2, -5/2, 0, 5/2, -2, 1/2]
    elif order == 6:
        coeffs = [1, -6, 15, -20, 15, -6, 1]
    else:
        print('UNSUPPORTED DIFFERENTIATION ORDER!')
    df = np.zeros_like(f)
    for ii in range(stride):
        coeff = coeffs[ii]
        if coeff != 0:
            df += coeff*np.roll(f,extra_pts-ii,axis=0)
    df = df/(dphi**order)
    return df[extra_pts::stride]

def dfCPT_dphi_dng(ngs, phis, fc, fj, phi_order, ng_order, n_charges = 2):
    # DOESN'T WORK FOR NG_ORDER=0!!!
    
    extra_phi_pts = math.ceil(phi_order/2)
    phi_stride = (2*extra_phi_pts)+1
    dphi = np.min(phis[1:]-np.roll(phis,1)[1:])/(10*extra_phi_pts)
    phi_interp = np.empty(phi_stride*len(phis),dtype=float)
    for ii in range(phi_stride):
        phi_interp[ii::phi_stride] = phis + ((ii-extra_phi_pts)*dphi)
    
    extra_ng_pts = math.ceil(ng_order/2)
    ng_stride = (2*extra_ng_pts)+1
    dng = np.min(ngs[1:]-np.roll(ngs,1)[1:])/(10*extra_ng_pts)
    ng_interp = np.empty(ng_stride*len(ngs),dtype=float)
    for jj in range(ng_stride):
        ng_interp[jj::ng_stride] = ngs + ((jj-extra_ng_pts)*dng)
        
    f = fCPT(ng_interp, phi_interp, fc, fj, n_charges)
    df = np.zeros_like(f)

    if phi_order == 1:
        phi_coeffs = [-1/2, 0, 1/2]
    elif phi_order == 2:
        phi_coeffs = [1, -2, 1]
    elif phi_order == 3:
        phi_coeffs = [-1/2, 1, 0, -1, 1/2]
    elif phi_order == 4:
        phi_coeffs = [1, -4, 6, -4, 1]
    elif phi_order == 5:
        phi_coeffs = [-1/2, 2, -5/2, 0, 5/2, -2, 1/2]
    elif phi_order == 6:
        phi_coeffs = [1, -6, 15, -20, 15, -6, 1]
    else:
        print('UNSUPPORTED DIFFERENTIATION ORDER!')

    for ii in range(phi_stride):
        coeff = phi_coeffs[ii]
        if coeff != 0:
            df += coeff*np.roll(f,extra_phi_pts-ii,axis=0)
    df = df/(dphi**phi_order)

    if ng_order == 1:
        ng_coeffs = [-1/2, 0, 1/2]
    elif ng_order == 2:
        ng_coeffs = [1, -2, 1]
    elif ng_order == 3:
        ng_coeffs = [-1/2, 1, 0, -1, 1/2]
    elif ng_order == 4:
        ng_coeffs = [1, -4, 6, -4, 1]
    elif ng_order == 5:
        ng_coeffs = [-1/2, 2, -5/2, 0, 5/2, -2, 1/2]
    elif ng_order == 6:
        ng_coeffs = [1, -6, 15, -20, 15, -6, 1]
    else:
        print('UNSUPPORTED DIFFERENTIATION ORDER!')

    dfdng = np.zeros_like(f)
    for ii in range(ng_stride):
        coeff = ng_coeffs[ii]
        if coeff != 0:
            dfdng += coeff*np.roll(df,extra_ng_pts-ii,axis=1)
    dfdng = dfdng/(dng**ng_order)
    return dfdng[extra_phi_pts::phi_stride,extra_ng_pts::ng_stride]

def dfCPT_dng(ngs, phi_exts, fc, fj, order, n_charges = 2):
    extra_pts = math.ceil(order/2)
    stride = (2*extra_pts)+1
    dng = np.min(ngs[1:]-np.roll(ngs,1)[1:])/(10*extra_pts)
    ng_interp = np.empty(stride*len(ngs),dtype=float)
    for ii in range(stride):
        ng_interp[ii::stride] = ngs + ((ii-extra_pts)*dng)
    f = fCPT(ng_interp, phi_exts, fc, fj, n_charges)
    if order == 1:
        coeffs = [-1/2, 0, 1/2]
    elif order == 2:
        coeffs = [1, -2, 1]
    elif order == 3:
        coeffs = [-1/2, 1, 0, -1, 1/2]
    elif order == 4:
        coeffs = [1, -4, 6, -4, 1]
    elif order == 5:
        coeffs = [-1/2, 2, -5/2, 0, 5/2, -2, 1/2]
    elif order == 6:
        coeffs = [1, -6, 15, -20, 15, -6, 1]
    else:
        print('UNSUPPORTED DIFFERENTIATION ORDER!')
    df = np.zeros_like(f)
    for ii in range(stride):
        coeff = coeffs[ii]
        if coeff != 0:
            df += coeff*np.roll(f,extra_pts-ii,axis=1)
    df = df/(dng**order)
    return df[:,extra_pts::stride]

def slopes_from_kerr():
    pass
