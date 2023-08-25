# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:10:36 2019

@author: Ben
"""
import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt



def ECPT_mathieu(m,ngs,Eratio):
    #ngs = array
    k = np.zeros_like(ngs)
    for l in [-1,1]:
        k += np.mod(np.rint(ngs+(l/2)),2)*(np.rint(ngs/2.0)+(l*((-1.0)**m)*((m+1)//2)))
    energies = spec.mathieu_a(ngs+(2*k),-0.5*Eratio)
    return energies

ratio = 1.0/3.0
ngs = np.linspace(-2.0,2.0,100)
E0 = ECPT_mathieu(0,ngs,ratio)
E1 = ECPT_mathieu(1,ngs,ratio)
E2 = ECPT_mathieu(2,ngs,ratio)

plt.plot(ngs,E0)
plt.plot(ngs,E1)
plt.plot(ngs,E2)
plt.show()