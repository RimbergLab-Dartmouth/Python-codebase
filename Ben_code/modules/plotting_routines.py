# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:05:54 2019

@author: Ben
"""

import numpy as np
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.image as image
from mpl_toolkits.axes_grid1 import ImageGrid


def compare_band_theory_experiment(phis, ngs, data_array, fit_array,\
                                   low_limit = 5.675E9,\
                                   high_limit = 5.825E9):
    fit_deviation = data_array-fit_array
    X,Y = np.meshgrid(phis,ngs)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xlabel = r'$\varphi_{\mathrm{ext}}$'
    ylabel = r'$n_{g}$'
    cbar_label = ''
    cmap1 = 'jet'
    cmap2 = 'viridis'
    scale = 12
    ticklabelsize = 14
    labelsize = 16
    plotlabelsize = 20
    f = plt.figure(figsize=(scale,scale))
    ax = f.gca(projection='3d')
    #cset = ax.contourf(X,Y,fit_deviation.T,zdir='z',offset=5.65E9,cmap=cmap2)
    surf = ax.plot_surface(X,Y,fit_array.T,cmap=cmap1,alpha=0.4,edgecolors='grey',linewidth=0.6)
    ax.scatter(X,Y,data_array.T,depthshade=True,marker='o',s=7.5,color='k')
    ax.set_zlim(low_limit,high_limit)
    ax.view_init(15,-65)
    plt.colorbar(surf,orientation='vertical',cmap=cmap1,fraction=0.035,pad=0.001,ax=ax)
    #plt.colorbar(cset,orientation='horizontal',cmap=cmap2,fraction=0.035,pad=0.001,ax=ax)
    plt.tight_layout()
    return f

def plot_band_surface(phis, ngs, array):
    X,Y = np.meshgrid(phis,ngs)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xlabel = r'$\varphi_{\mathrm{ext}}$'
    ylabel = r'$n_{g}$'
    cbar_label = ''
    cmap1 = 'jet'
    cmap2 = 'viridis'
    scale = 12
    ticklabelsize = 14
    labelsize = 16
    plotlabelsize = 20
    f = plt.figure(figsize=(scale,scale))
    ax = f.gca(projection='3d')
    surf = ax.plot_surface(X,Y,array.T,cmap=cmap1,alpha=1.0,edgecolors='grey',linewidth=0.6)
    #ax.set_zlim(5.675E9,5.825E9)
    #ax.view_init(30,-40)
    ax.view_init(15,-65)
    plt.colorbar(surf,orientation='vertical',cmap=cmap1,fraction=0.035,pad=0.001,ax=ax)
    #plt.colorbar(cset,orientation='horizontal',cmap=cmap2,fraction=0.035,pad=0.001,ax=ax)
    plt.tight_layout()
    return f

def plot_band_colormap(phis, ngs, array, \
                        cmap_str = 'inferno', \
                        title = None, \
                        cbar_label = '', \
                        size_params = None,\
                        cbar_lims = None,
                        use_edges = True):
    phis = phis/np.pi
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xlabel = r'$\Phi_{\mathrm{ext}}/\Phi_{0}$'
    ylabel = r'$n_{g}$'
    phi_axis = np.array(phis)
    ng_axis = np.array(ngs)
    if use_edges:
        dphi = np.mean((phis-np.roll(phis,1))[1:])
        phi0 = phis[0]-dphi
        phi_axis = np.hstack((phi0,phis))
        phi_axis += dphi/2.0
        dng = np.mean((ngs-np.roll(ngs,1))[1:])
        ng0 = ngs[0]-dng
        ng_axis = np.hstack((ng0,ngs))
        ng_axis += dng/2.0
    if isinstance(size_params,type(None)):
        scale = 7
        ticklabelsize = 14
        labelsize = 16
        plotlabelsize = 20
    else:
        scale,ticklabelsize,labelsize,plotlabelsize = size_params
    f = pylab.figure(figsize=(scale,scale))
    if not isinstance(title,type(None)):
        pylab.title(title,fontsize=plotlabelsize)
    if isinstance(cbar_lims,type(None)):
        pylab.pcolormesh(phi_axis,ng_axis,array.transpose(),\
                         cmap=cmap_str)
    else:
        pylab.pcolormesh(phi_axis,ng_axis,array.transpose(),\
                         cmap=cmap_str,\
                         vmin = cbar_lims[0],\
                         vmax = cbar_lims[1])
    cbar = pylab.colorbar()
    cbar.ax.tick_params(labelsize = ticklabelsize)
    cbar.set_label(label=cbar_label,size=labelsize)
    pylab.xlabel(xlabel,fontsize=labelsize)
    pylab.ylabel(ylabel,fontsize=labelsize)
    pylab.tick_params(labelsize=ticklabelsize)
    return f

def plot_delta_epsilon_colormap(deltas, epsilons, array, \
                        cmap_str = 'inferno', \
                        title = '', \
                        cbar_label = '', \
                        size_params = None,\
                        cbar_lims = None,
                        use_edges = True):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xlabel = r'$\delta$'
    ylabel = r'$\epsilon$'
    delta_axis = np.array(deltas)
    epsilon_axis = np.array(epsilons)
    if use_edges:
        ddelta = np.mean((deltas-np.roll(deltas,1))[1:])
        delta0 = deltas[0]-ddelta
        delta_axis = np.hstack((delta0,deltas))
        delta_axis += ddelta/2.0
        depsilon = np.mean((epsilons-np.roll(epsilons,1))[1:])
        epsilon0 = epsilons[0]-depsilon
        epsilon_axis = np.hstack((epsilon0,epsilons))
        epsilon_axis += depsilon/2.0
    if isinstance(size_params,type(None)):
        scale = 7
        ticklabelsize = 14
        labelsize = 16
        plotlabelsize = 20
    else:
        scale,ticklabelsize,labelsize,plotlabelsize = size_params
    f = pylab.figure(figsize=(scale,scale))
    pylab.title(title,fontsize=plotlabelsize)
    if isinstance(cbar_lims,type(None)):
        pylab.pcolormesh(delta_axis,epsilon_axis,array.transpose(),\
                         cmap=cmap_str)
    else:
        pylab.pcolormesh(delta_axis,epsilon_axis,array.transpose(),\
                         cmap=cmap_str,\
                         vmin = cbar_lims[0],\
                         vmax = cbar_lims[1])
    cbar = pylab.colorbar()
    cbar.ax.tick_params(labelsize = ticklabelsize)
    cbar.set_label(label=cbar_label,size=labelsize)
    pylab.xlabel(xlabel,fontsize=labelsize)
    pylab.ylabel(ylabel,fontsize=labelsize)
    pylab.tick_params(labelsize=ticklabelsize)
    return f

def plot_colormap(xs, ys, array, \
                  xlabel = r'x',
                  ylabel = r'y',
                cmap_str = 'inferno', \
                title = '', \
                cbar_label = '', \
                size_params = None,\
                cbar_lims = None,
                use_edges = True):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    x_axis = np.array(xs)
    y_axis = np.array(ys)
    if use_edges:
        dx = np.mean((xs-np.roll(xs,1))[1:])
        x0 = xs[0]-dx
        x_axis = np.hstack((x0,xs))
        x_axis += dx/2.0
        dy = np.mean((ys-np.roll(ys,1))[1:])
        y0 = ys[0]-dy
        y_axis = np.hstack((y0,ys))
        y_axis += dy/2.0
    if isinstance(size_params,type(None)):
        scale = 7
        ticklabelsize = 14
        labelsize = 16
        plotlabelsize = 20
    else:
        scale,ticklabelsize,labelsize,plotlabelsize = size_params
    f = pylab.figure(figsize=(scale,scale))
    pylab.title(title,fontsize=plotlabelsize)
    if isinstance(cbar_lims,type(None)):
        pylab.pcolormesh(x_axis,y_axis,array.transpose(),\
                         cmap=cmap_str)
    else:
        pylab.pcolormesh(x_axis,y_axis,array.transpose(),\
                         cmap=cmap_str,\
                         vmin = cbar_lims[0],\
                         vmax = cbar_lims[1])
    cbar = pylab.colorbar()
    cbar.ax.tick_params(labelsize = ticklabelsize)
    cbar.set_label(label=cbar_label,size=labelsize)
    pylab.xlabel(xlabel,fontsize=labelsize)
    pylab.ylabel(ylabel,fontsize=labelsize)
    pylab.tick_params(labelsize=ticklabelsize)
    return f

def compare_band_colormaps_old(phis, ngs, data1, data2,
                           cmap_str = 'inferno', 
                           title = '', 
                           cbar_label = '', 
                           size_params = None,
                           use_edges = True):
    phis = phis/np.pi
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xlabel = r'$\Phi_{\mathrm{ext}}/\Phi_{0}$'
    ylabel = r'$n_{g}$'
    phi_axis = np.array(phis)
    ng_axis = np.array(ngs)
    if use_edges:
        dphi = np.mean((phis-np.roll(phis,1))[1:])
        phi0 = phis[0]-dphi
        phi_axis = np.hstack((phi0,phis))
        phi_axis += dphi/2.0
        dng = np.mean((ngs-np.roll(ngs,1))[1:])
        ng0 = ngs[0]-dng
        ng_axis = np.hstack((ng0,ngs))
        ng_axis += dng/2.0
    if isinstance(size_params,type(None)):
        scale = 7
        ticklabelsize = 14
        labelsize = 16
        plotlabelsize = 20
    else:
        scale,ticklabelsize,labelsize,plotlabelsize = size_params
    
    
    cbar_lims = [min(np.amin(data1),np.amin(data2)),
                 max(np.amax(data1),np.amax(data2))]
        
    fig,axarr = plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(2*scale,scale))


    axarr[0].pcolormesh(phi_axis,ng_axis,data1.transpose(),
                         cmap=cmap_str,
                         vmin = cbar_lims[0],
                         vmax = cbar_lims[1])
    im=axarr[1].pcolormesh(phi_axis,ng_axis,data2.transpose(),
                         cmap=cmap_str,
                         vmin = cbar_lims[0],
                         vmax = cbar_lims[1])
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im,ax=cbar_ax)
    cbar.ax.tick_params(labelsize = ticklabelsize)
    cbar.set_label(label=cbar_label,size=labelsize)
    for ii in range(2):
        axarr[ii].set_xlabel(xlabel,fontsize=labelsize)
        axarr[ii].set_ylabel(ylabel,fontsize=labelsize)
        axarr[ii].tick_params(labelsize=ticklabelsize)
    return fig

def compare_band_colormaps(phis, ngs, data1, data2,
                           cmap_str = 'inferno', 
                           title = '', 
                           cbar_label = '', 
                           cbar_units = '',
                           size_params = None,
                           use_edges = True,
                           pad = 0.35,
                           cbar_lims = None,
                           xtick_locs = None,
                           xtick_labels = None,
                           ytick_locs = None,
                           ytick_labels = None,
                           cbar_tick_locs = None,
                           cbar_tick_labels = None,
                           suptitle = None,
                           subtitles = None,
                           rotate_labels = False,
                           xlabelpad = None,
                           ylabelpad = None,
                           cbarlabelpad = None,
                           aspect = 0.5):
    phis = phis/np.pi
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xlabel = r'$\Phi_{\mathrm{ext}}/\Phi_{0}$'
    ylabel = r'$n_{g}$'
    phi_axis = np.array(phis)
    ng_axis = np.array(ngs)
    if use_edges:
        dphi = np.mean((phis-np.roll(phis,1))[1:])
        phi0 = phis[0]-dphi
        phi_axis = np.hstack((phi0,phis))
        phi_axis += dphi/2.0
        dng = np.mean((ngs-np.roll(ngs,1))[1:])
        ng0 = ngs[0]-dng
        ng_axis = np.hstack((ng0,ngs))
        ng_axis += dng/2.0
    if isinstance(size_params,type(None)):
        width = 14
        ticklabelsize = 16
        labelsize = 20
        plotlabelsize = 20
        subtitlesize = 24
        suptitlesize = 28
    else:
        width,ticklabelsize,labelsize,plotlabelsize,subtitlesize,suptitlesize = size_params
    
    if isinstance(cbar_lims,type(None)):
        cbar_lims = [min(np.amin(data1),np.amin(data2)),
                     max(np.amax(data1),np.amax(data2))]
    
    height = aspect*width
    fig = plt.figure(figsize=(width,height))
    axarr = ImageGrid(fig, 111,          # as in plt.subplot(111)
                     nrows_ncols=(1,2),
                     axes_pad=pad,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=pad,
                     )
    

    axarr[0].pcolormesh(phi_axis,ng_axis,data1.transpose(),
                         cmap=cmap_str,
                         vmin = cbar_lims[0],
                         vmax = cbar_lims[1])
    im=axarr[1].pcolormesh(phi_axis,ng_axis,data2.transpose(),
                         cmap=cmap_str,
                         vmin = cbar_lims[0],
                         vmax = cbar_lims[1])
    
    if isinstance(cbar_tick_locs,type(None)):
        cbar = axarr[1].cax.colorbar(im)
    else:
        cbar = axarr[1].cax.colorbar(im,ticks=cbar_tick_locs)
        cbar.ax.set_yticklabels(cbar_tick_labels)
    axarr[1].cax.toggle_label(True)
    for ii in range(2):
        axarr[ii].set_xlabel(xlabel,fontsize=labelsize)
        if rotate_labels:
            axarr[ii].set_ylabel(ylabel,rotation=0,fontsize=labelsize)
        else:
            axarr[ii].set_ylabel(ylabel,fontsize=labelsize)
        axarr[ii].tick_params(labelsize=ticklabelsize)

    if rotate_labels:
        final_cbar_label = cbar_label+r'\\'+cbar_units
        cbar.set_label_text(final_cbar_label,size=labelsize,rotation=0)
        cbar.ax.tick_params(labelsize=ticklabelsize)
    else:
        final_cbar_label = cbar_label+r' '+cbar_units
        cbar.set_label_text(final_cbar_label,size=labelsize)
        cbar.ax.tick_params(labelsize=ticklabelsize)    
    
    if not isinstance(cbarlabelpad,type(None)):
        cbar.ax.get_yaxis().labelpad = cbarlabelpad
    
    if not isinstance(xtick_locs,type(None)):
        axarr[0].set_xticks(xtick_locs)
        axarr[1].set_xticks(xtick_locs)
    if not isinstance(ytick_locs,type(None)):
        axarr[0].set_yticks(ytick_locs)
        axarr[1].set_yticks(ytick_locs)
    if not isinstance(xtick_labels,type(None)):
        axarr[0].set_xticklabels(xtick_labels)
        axarr[1].set_xticklabels(xtick_labels)
    if not isinstance(ytick_labels,type(None)):
        axarr[0].set_yticklabels(ytick_labels)
        axarr[1].set_yticklabels(ytick_labels)
        
    if not isinstance(suptitle,type(None)):
        fig.suptitle(suptitle,fontsize=suptitlesize)
    if not isinstance(subtitles,type(None)):
        axarr[0].set_title(subtitles[0],fontsize=subtitlesize)
        axarr[1].set_title(subtitles[1],fontsize=subtitlesize)
    
    
    return fig

def plot_band_image(phis, ngs, array, \
                        cmap_str = 'inferno', \
                        title = '', \
                        cbar_label = '', \
                        size_params = None,\
                        cbar_lims = None):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xlabel = r'$\varphi_{\mathrm{ext}}$'
    ylabel = r'$n_{g}$'
    if isinstance(size_params,type(None)):
        scale = 7
        ticklabelsize = 14
        labelsize = 16
        plotlabelsize = 20
    else:
        scale,ticklabelsize,labelsize,plotlabelsize = size_params
        
    fig, ax = plt.subplots(figsize=(scale,scale))
    #ax = fig.axes
    plt.title(title,fontsize=plotlabelsize)
    extent = (phis[0],phis[-1],ngs[0],ngs[-1])
    if isinstance(cbar_lims,type(None)):
        im = image.NonUniformImage(ax,
                                   interpolation='nearest',
                                   extent=extent,
                                   cmap=cmap_str)
    else:
        im = image.NonUniformImage(ax,
                                   interpolation='nearest',
                                   extent=extent,
                                   cmap=cmap_str,
                                   vmin = cbar_lims[0],\
                                   vmax = cbar_lims[1])
    im.set_data(phis,ngs,array.T)
    ax.images.append(im)
    ax.set_xlim(extent[0],extent[1])
    ax.set_ylim(extent[2],extent[3])
    cbar = fig.colorbar(im,ax=ax)
    cbar.ax.tick_params(labelsize = ticklabelsize)
    cbar.set_label(label=cbar_label,size=labelsize)
    plt.xlabel(xlabel,fontsize=labelsize)
    plt.ylabel(ylabel,fontsize=labelsize)
    plt.tick_params(labelsize=ticklabelsize)
    return fig


def compare_band_colormap_fits(phis, ngs, data_array, fit_array, \
                        cmap_str = 'inferno', \
                        title = '', \
                        labels = None, \
                        size_params = None,\
                        cbar_label = '',
                        cbar_pad = 1.0):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # aspect ratio
    asp = (phis[-1]-phis[0])/(ngs[-1]-ngs[0])
    
    if isinstance(labels,type(None)):
        xlabel = r'$\varphi_{\mathrm{ext}}$'
        ylabel = r'$n_{g}$'
    else:
        xlabel,ylabel = labels
    
    if isinstance(size_params,type(None)):
        scale = 7
        ticklabelsize = 14
        labelsize = 16
        plotlabelsize = 20
    else:
        scale,ticklabelsize,labelsize,plotlabelsize = size_params
        
    cbar_lims = [np.amin(np.hstack((data_array,fit_array))), np.amax(np.hstack((data_array,fit_array)))]
    """
    f,axarr = plt.subplots(1,2,figsize=((2*scale),scale))
    #axarr[0].title(title,fontsize=plotlabelsize)
    axarr[0].pcolormesh(phis,ngs,data_array.transpose(),\
                         cmap=cmap_str,\
                         vmin = cbar_lims[0],\
                         vmax = cbar_lims[1])
    axarr[0].set_xlabel(xlabel,fontsize=labelsize)
    axarr[0].set_ylabel(ylabel,fontsize=labelsize)
    axarr[0].tick_params(labelsize=ticklabelsize)
    #axarr[0].set_aspect(asp)
    cbar_map = axarr[1].pcolormesh(phis,ngs,fit_array.transpose(),\
                         cmap=cmap_str,\
                         vmin = cbar_lims[0],\
                         vmax = cbar_lims[1])
    axarr[1].set_xlabel(xlabel,fontsize=labelsize)
    axarr[1].tick_params(labelsize=ticklabelsize)
    #axarr[1].set_aspect(asp)
    cbar_axes = f.add_subplot(133)
    cbar = f.colorbar(cbar_map,ax=cbar_axes)
    cbar.ax.tick_params(labelsize = ticklabelsize)
    cbar.set_label(label=cbar_label,size=labelsize)
    #plt.tight_layout()
    """
    f = plt.figure()
    ax0 = plt.subplot()
    #axarr[0].title(title,fontsize=plotlabelsize)
    axarr[0].pcolormesh(phis,ngs,data_array.transpose(),\
                         cmap=cmap_str,\
                         vmin = cbar_lims[0],\
                         vmax = cbar_lims[1])
    axarr[0].set_xlabel(xlabel,fontsize=labelsize)
    axarr[0].set_ylabel(ylabel,fontsize=labelsize)
    axarr[0].tick_params(labelsize=ticklabelsize)
    #axarr[0].set_aspect(asp)
    cbar_map = axarr[1].pcolormesh(phis,ngs,fit_array.transpose(),\
                         cmap=cmap_str,\
                         vmin = cbar_lims[0],\
                         vmax = cbar_lims[1])
    axarr[1].set_xlabel(xlabel,fontsize=labelsize)
    axarr[1].tick_params(labelsize=ticklabelsize)
    #axarr[1].set_aspect(asp)
    cbar_axes = f.add_subplot(133)
    cbar = f.colorbar(cbar_map,ax=cbar_axes)
    cbar.ax.tick_params(labelsize = ticklabelsize)
    cbar.set_label(label=cbar_label,size=labelsize)
    #plt.tight_layout()
    return f

def plot_reflection_coefficient_fits(freqs, gamma_data, gamma_fit, \
                                     size_params=None,\
                                     text_above = None,\
                                     text_below = None,\
                                     height_padding = 0.0,
                                     equal_aspect = True):
    # visualize data & fit for arbitrary fitting procedure
    # feed in xfit and yfit rather than parameters from a specific model

    freqs = freqs/(1.0E9)
        
    if isinstance(size_params,type(None)):
        scale = 5
        ticklabelsize = 14
        labelsize = 16
        plotlabelsize = 20
    else:
        scale,ticklabelsize,labelsize,plotlabelsize = size_params
    
    
    f,axarr = plt.subplots(1,3,figsize=(3*scale, scale+height_padding))
    xdata = np.real(gamma_data)
    ydata = np.imag(gamma_data)
    xfit = np.real(gamma_fit)
    yfit = np.imag(gamma_fit)

    linmag = np.sqrt((xdata**2)+(ydata**2))
    phase = (180.0/np.pi)*np.arctan2(ydata,xdata)
    linmag_fit = np.sqrt((xfit**2)+(yfit**2))
    phase_fit = (180.0/np.pi)*np.arctan2(yfit, xfit)
    
    if not isinstance(text_above,type(None)):
        axarr[0].text(0.0, 1.2, text_above, transform=axarr[0].transAxes, \
                     fontsize=plotlabelsize, fontweight='bold', va='bottom')
        
    if not isinstance(text_below,type(None)):
        axarr[0].text(0.0, -0.3, text_below, transform=axarr[0].transAxes, \
                     fontsize=plotlabelsize, fontweight='bold', va='top')
    
    axarr[0].plot(xdata,ydata,linestyle='None',marker='.')
    axarr[0].plot(xfit,yfit)
    axarr[0].set_xlabel('Re[$S_{11}$]',fontsize=labelsize)
    axarr[0].set_ylabel('Im[$S_{11}$]',fontsize=labelsize)
    axarr[0].tick_params(labelsize=ticklabelsize)
    if equal_aspect:
        axarr[0].set_aspect('equal')
    
    axarr[1].plot(freqs,linmag,linestyle='None',marker='.')
    axarr[1].plot(freqs,linmag_fit)
    axarr[1].set_xlabel('Frequency (GHz)',fontsize=labelsize)
    axarr[1].set_ylabel('$| S_{11} |$',fontsize=labelsize)
    axarr[1].tick_params(labelsize=ticklabelsize)

    axarr[2].plot(freqs,phase,linestyle='None',marker='.')
    axarr[2].plot(freqs,phase_fit)
    axarr[2].set_xlabel('Frequency (GHz)',fontsize=labelsize)
    axarr[2].set_ylabel(r'$\mathrm{arg}\left[S_{11}\right]\;\mathrm{(degrees)}$',fontsize=labelsize)
    axarr[2].tick_params(labelsize=ticklabelsize)
        
    plt.tight_layout()
    return f

def plot_reflection_coefficient_fits_ktot(freqs, gamma_data, gamma_fit, 
                                          ktot_freqs, ktot_gammas, 
                                             size_params=None,\
                                             text_above = None,\
                                             text_below = None,\
                                             height_padding = 0.0,
                                             equal_aspect = True):
    # visualize data & fit for arbitrary fitting procedure
    # feed in xfit and yfit rather than parameters from a specific model

    freqs = freqs/(1.0E9)
    ktot_freqs = ktot_freqs/(1.0E9)
    
    data_ktot_freqs = np.zeros_like(ktot_freqs)
    data_ktot_gammas = np.zeros_like(ktot_gammas)
    for ii in range(len(ktot_freqs)):
        ind = np.argmin(np.abs(freqs-ktot_freqs[ii]))
        data_ktot_freqs[ii] = freqs[ind]
        data_ktot_gammas[ii] = gamma_data[ind]
    ktot_freqs = np.hstack((ktot_freqs,data_ktot_freqs))
    ktot_gammas = np.hstack((ktot_gammas,data_ktot_gammas))
        
    if isinstance(size_params,type(None)):
        scale = 5
        ticklabelsize = 14
        labelsize = 16
        plotlabelsize = 20
    else:
        scale,ticklabelsize,labelsize,plotlabelsize = size_params
    
    markersize = 10
    
    
    f,axarr = plt.subplots(1,3,figsize=(3*scale, scale+height_padding))
    xdata = np.real(gamma_data)
    ydata = np.imag(gamma_data)
    xfit = np.real(gamma_fit)
    yfit = np.imag(gamma_fit)
    ktot_xs = np.real(ktot_gammas)
    ktot_ys = np.imag(ktot_gammas)

    linmag = np.sqrt((xdata**2)+(ydata**2))
    phase = (180.0/np.pi)*np.arctan2(ydata,xdata)
    linmag_fit = np.sqrt((xfit**2)+(yfit**2))
    phase_fit = (180.0/np.pi)*np.arctan2(yfit, xfit)
    ktot_linmags = np.sqrt((ktot_xs**2)+(ktot_ys**2))
    ktot_phases = (180.0/np.pi)*np.arctan2(ktot_ys,ktot_xs)
    
    if not isinstance(text_above,type(None)):
        axarr[0].text(0.0, 1.2, text_above, transform=axarr[0].transAxes, \
                     fontsize=plotlabelsize, fontweight='bold', va='bottom')
        
    if not isinstance(text_below,type(None)):
        axarr[0].text(0.0, -0.3, text_below, transform=axarr[0].transAxes, \
                     fontsize=plotlabelsize, fontweight='bold', va='top')
    
    axarr[0].plot(xdata,ydata,linestyle='None',marker='.')
    axarr[0].plot(xfit,yfit)
    axarr[0].plot(ktot_xs,ktot_ys,'o',color='r',fillstyle='none',markersize=markersize)
    axarr[0].set_xlabel('Re[$S_{11}$]',fontsize=labelsize)
    axarr[0].set_ylabel('Im[$S_{11}$]',fontsize=labelsize)
    axarr[0].tick_params(labelsize=ticklabelsize)
    if equal_aspect:
        axarr[0].set_aspect('equal')
    
    axarr[1].plot(freqs,linmag,linestyle='None',marker='.')
    axarr[1].plot(freqs,linmag_fit)
    axarr[1].plot(ktot_freqs,ktot_linmags,'o',color='r',fillstyle='none',markersize=markersize)
    axarr[1].set_xlabel('Frequency (GHz)',fontsize=labelsize)
    axarr[1].set_ylabel('$| S_{11} |$',fontsize=labelsize)
    axarr[1].tick_params(labelsize=ticklabelsize)

    axarr[2].plot(freqs,phase,linestyle='None',marker='.')
    axarr[2].plot(freqs,phase_fit)
    axarr[2].plot(ktot_freqs,ktot_phases,'o',color='r',fillstyle='none',markersize=markersize)
    axarr[2].set_xlabel('Frequency (GHz)',fontsize=labelsize)
    axarr[2].set_ylabel(r'$\mathrm{arg}\left[S_{11}\right]\;\mathrm{(degrees)}$',fontsize=labelsize)
    axarr[2].tick_params(labelsize=ticklabelsize)
        
    plt.tight_layout()
    return f

def plot_reflection_coefficient(freqs, 
                                gamma, 
                                lines=False, 
                                scale=5,
                                equal_aspect=True):
    # visualize complex reflection coefficient
    x = np.real(gamma)
    y = np.imag(gamma)
    linmag = np.sqrt((x**2)+(y**2))
    phase = (180.0/np.pi)*np.arctan2(y,x)
    
    f,axarr = plt.subplots(1,3,figsize=(3*scale, scale))
    if lines:
        axarr[0].plot(x,y)
    else:
        axarr[0].plot(x,y,linestyle='None',marker='.')
    axarr[0].set(xlabel='Re[S11]',ylabel='Im[S11]')
    if equal_aspect:
        axarr[0].set_aspect('equal')
    
    if lines:
        axarr[1].plot(freqs,linmag)
    else:
        axarr[1].plot(freqs,linmag,linestyle='None',marker='.')
    axarr[1].set(xlabel='Frequency (Hz)',ylabel='|S11|')
    #axarr[1].set_aspect(1.0)
    
    if lines:
        axarr[2].plot(freqs,phase)
    else:
        axarr[2].plot(freqs,phase,linestyle='None',marker='.')
    axarr[2].set(xlabel='Frequency (Hz)',ylabel='arg[S11] (degrees)')
    plt.tight_layout()
    return f

def compare_reflection_coefficient_fits_ktot(freqs, 
                                             gamma_data, 
                                             gamma_fit, 
                                             gamma_fit2,
                                             ktot_freqs, 
                                             ktot_freqs2,
                                             ktot_gammas, 
                                             ktot_gammas2,
                                             size_params=None,
                                             text_above = None,
                                             text_below = None,
                                             height_padding = 0.0,
                                             equal_aspect = True):
    # visualize data & fit for arbitrary fitting procedure
    # feed in xfit and yfit rather than parameters from a specific model

    freqs = freqs/(1.0E9)
    ktot_freqs = ktot_freqs/(1.0E9)
    ktot_freqs2 = ktot_freqs2/(1.0E9)
    
    data_ktot_freqs = np.zeros_like(ktot_freqs)
    data_ktot_gammas = np.zeros_like(ktot_gammas)
    data_ktot_freqs2 = np.zeros_like(ktot_freqs2)
    data_ktot_gammas2 = np.zeros_like(ktot_gammas2)
    for ii in range(len(ktot_freqs)):
        ind = np.argmin(np.abs(freqs-ktot_freqs[ii]))
        data_ktot_freqs[ii] = freqs[ind]
        data_ktot_gammas[ii] = gamma_data[ind]
        ind = np.argmin(np.abs(freqs-ktot_freqs2[ii]))
        data_ktot_freqs2[ii] = freqs[ind]
        data_ktot_gammas2[ii] = gamma_data[ind]
    ktot_freqs = np.hstack((ktot_freqs,data_ktot_freqs))
    ktot_gammas = np.hstack((ktot_gammas,data_ktot_gammas))
    ktot_freqs2 = np.hstack((ktot_freqs2,data_ktot_freqs2))
    ktot_gammas2 = np.hstack((ktot_gammas2,data_ktot_gammas2))
    
    if isinstance(size_params,type(None)):
        scale = 5
        ticklabelsize = 14
        labelsize = 16
        plotlabelsize = 20
    else:
        scale,ticklabelsize,labelsize,plotlabelsize = size_params
    
    markersize = 10
    
    
    f,axarr = plt.subplots(1,3,figsize=(3*scale, scale+height_padding))
    xdata = np.real(gamma_data)
    ydata = np.imag(gamma_data)
    xfit = np.real(gamma_fit)
    yfit = np.imag(gamma_fit)
    xfit2 = np.real(gamma_fit2)
    yfit2 = np.imag(gamma_fit2)
    ktot_xs = np.real(ktot_gammas)
    ktot_ys = np.imag(ktot_gammas)
    ktot_xs2 = np.real(ktot_gammas2)
    ktot_ys2 = np.imag(ktot_gammas2)

    linmag = np.sqrt((xdata**2)+(ydata**2))
    phase = (180.0/np.pi)*np.arctan2(ydata,xdata)
    linmag_fit = np.sqrt((xfit**2)+(yfit**2))
    phase_fit = (180.0/np.pi)*np.arctan2(yfit, xfit)
    linmag_fit2 = np.sqrt((xfit2**2)+(yfit2**2))
    phase_fit2 = (180.0/np.pi)*np.arctan2(yfit2, xfit2)
    ktot_linmags = np.sqrt((ktot_xs**2)+(ktot_ys**2))
    ktot_phases = (180.0/np.pi)*np.arctan2(ktot_ys,ktot_xs)
    ktot_linmags2 = np.sqrt((ktot_xs2**2)+(ktot_ys2**2))
    ktot_phases2 = (180.0/np.pi)*np.arctan2(ktot_ys2,ktot_xs2)
    
    if not isinstance(text_above,type(None)):
        axarr[0].text(0.0, 1.2, text_above, transform=axarr[0].transAxes, \
                     fontsize=plotlabelsize, fontweight='bold', va='bottom')
        
    if not isinstance(text_below,type(None)):
        axarr[0].text(0.0, -0.3, text_below, transform=axarr[0].transAxes, \
                     fontsize=plotlabelsize, fontweight='bold', va='top')
    
    axarr[0].plot(xdata,ydata,linestyle='None',marker='.')
    axarr[0].plot(xfit,yfit)
    axarr[0].plot(xfit2,yfit2)
    axarr[0].plot(ktot_xs,ktot_ys,'o',color='r',fillstyle='none',markersize=markersize)
    axarr[0].plot(ktot_xs2,ktot_ys2,'o',color='g',fillstyle='none',markersize=markersize)
    axarr[0].set_xlabel('Re[$S_{11}$]',fontsize=labelsize)
    axarr[0].set_ylabel('Im[$S_{11}$]',fontsize=labelsize)
    axarr[0].tick_params(labelsize=ticklabelsize)
    if equal_aspect:
        axarr[0].set_aspect('equal')
    
    axarr[1].plot(freqs,linmag,linestyle='None',marker='.')
    axarr[1].plot(freqs,linmag_fit)
    axarr[1].plot(freqs,linmag_fit2)
    axarr[1].plot(ktot_freqs,ktot_linmags,'o',color='r',fillstyle='none',markersize=markersize)
    axarr[1].plot(ktot_freqs2,ktot_linmags2,'o',color='g',fillstyle='none',markersize=markersize)
    axarr[1].set_xlabel('Frequency (GHz)',fontsize=labelsize)
    axarr[1].set_ylabel('$| S_{11} |$',fontsize=labelsize)
    axarr[1].tick_params(labelsize=ticklabelsize)

    axarr[2].plot(freqs,phase,linestyle='None',marker='.')
    axarr[2].plot(freqs,phase_fit)
    axarr[2].plot(freqs,phase_fit2)
    axarr[2].plot(ktot_freqs,ktot_phases,'o',color='r',fillstyle='none',markersize=markersize)
    axarr[2].plot(ktot_freqs2,ktot_phases2,'o',color='g',fillstyle='none',markersize=markersize)
    axarr[2].set_xlabel('Frequency (GHz)',fontsize=labelsize)
    axarr[2].set_ylabel(r'$\mathrm{arg}\left[S_{11}\right]\;\mathrm{(degrees)}$',fontsize=labelsize)
    axarr[2].tick_params(labelsize=ticklabelsize)
        
    plt.tight_layout()
    return f

def compare_reflection_coefficient_fits(freqs, gamma_data, \
                                                gamma_fit, \
                                                gamma_data2 = None, \
                                                gamma_fit2 = None, \
                                                size_params = None, \
                                                labels = None):
    # visualize data & fit for two different fitting procedures
    # feed in xfit and yfit rather than parameters from a specific model
    
    freqs = freqs/(1.0E9)
    
    if isinstance(labels,type(None)):
        labels = [r'\textbf{(a)}',r'\textbf{(b)}']
    
    if isinstance(size_params,type(None)):
        scale = 12
        ticklabelsize = 14
        labelsize = 16
        plotlabelsize = 20
    else:
        scale,ticklabelsize,labelsize,plotlabelsize = size_params
    
    if isinstance(gamma_data2,type(None)):
        gamma_data2 = gamma_data
    if isinstance(gamma_fit2,type(None)):
        gamma_fit2 = gamma_fit

    
    f,axarr = plt.subplots(2,3,figsize=(3*scale, 2*scale))
    for ii in range(2):
        label = labels[ii]
        if ii == 0:
            xdata = np.real(gamma_data)
            ydata = np.imag(gamma_data)
            xfit = np.real(gamma_fit)
            yfit = np.imag(gamma_fit)
        else:
            xdata = np.real(gamma_data2)
            ydata = np.imag(gamma_data2)
            xfit = np.real(gamma_fit2)
            yfit = np.imag(gamma_fit2)
        linmag = np.sqrt((xdata**2)+(ydata**2))
        phase = (180.0/np.pi)*np.arctan2(ydata,xdata)
        linmag_fit = np.sqrt((xfit**2)+(yfit**2))
        phase_fit = (180.0/np.pi)*np.arctan2(yfit, xfit)
            
        axarr[ii,0].text(-0.2, 1.1, label, transform=axarr[ii,0].transAxes, \
                         fontsize=plotlabelsize, fontweight='bold', va='top', \
                         ha = 'left')
        
        axarr[ii,0].plot(xdata,ydata,linestyle='None',marker='.')
        axarr[ii,0].plot(xfit,yfit)
        axarr[ii,0].set_xlabel('Re[$S_{11}$]',fontsize=labelsize)
        axarr[ii,0].set_ylabel('Im[$S_{11}$]',fontsize=labelsize)
        #axarr[ii,0].set_aspect('equal')
        axarr[ii,0].tick_params(labelsize=ticklabelsize)
        
        axarr[ii,1].plot(freqs,linmag,linestyle='None',marker='.')
        axarr[ii,1].plot(freqs,linmag_fit)
        axarr[ii,1].set_xlabel('Frequency (GHz)',fontsize=labelsize)
        axarr[ii,1].set_ylabel('$| S_{11} |$',fontsize=labelsize)
        axarr[ii,1].tick_params(labelsize=ticklabelsize)

        axarr[ii,2].plot(freqs,phase,linestyle='None',marker='.')
        axarr[ii,2].plot(freqs,phase_fit)
        axarr[ii,2].set_xlabel('Frequency (GHz)',fontsize=labelsize)
        axarr[ii,2].set_ylabel(r'$\mathrm{arg}\left[S_{11}\right]\;\mathrm{(degrees)}$',fontsize=labelsize)
        axarr[ii,2].tick_params(labelsize=ticklabelsize)
        
    plt.tight_layout()
    return f
