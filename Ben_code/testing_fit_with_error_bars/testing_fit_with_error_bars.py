# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:39:06 2019

@author: Ben
"""

import numpy as np
import lmfit
import matplotlib.pyplot as plt

n_points = 100

actual_slope = 7.0
actual_intercept = -3.0

noise_scale = 1.0
noise_samples = np.random.normal(0.0,noise_scale,n_points)

err_scale = 1.0
yerr = np.ones(n_points)*err_scale
#yerr = np.linspace(0.01,1,n_points)*err_scale
#yerr = np.random.normal(0.0,noise_scale,n_points)
#yerr = noise_samples

x = np.linspace(0,1,n_points)
y = (actual_slope*x) + actual_intercept + noise_samples
weights = (1/yerr)*(1/np.sum(1/yerr))

def line(x,slope,intercept):
    result = (slope*x)+intercept
    return result

model = lmfit.Model(line)
params = model.make_params()
params['slope'].value = actual_slope
params['intercept'].value = actual_intercept

print('Fit without weights:')
result = model.fit(y,params=params,x=x)
print(result.fit_report())
print(result.ci_report())

print('Fit with weights:')
result = model.fit(y,params=params,x=x,weights=weights)
c_intervals = result.conf_interval()
print(c_intervals)
yfit = result.best_fit
print(result.fit_report())
print(result.ci_report())


labelsize = 24
ticklabelsize = 20
markersize = 8
linewidth = 3

fig,ax = plt.subplots(figsize=(12,8))
ax.errorbar(x,y,yerr=yerr,fmt='o',capsize=10)
ax.plot(x,yfit,linewidth=linewidth)
ax.set_xlabel('X',fontsize=labelsize)
ax.set_ylabel('Y',fontsize=labelsize)
ax.tick_params(labelsize=ticklabelsize)
ax.set_title('Noisy Line and Best Fit',fontsize=labelsize)
plt.show()