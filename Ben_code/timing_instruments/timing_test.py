# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:37:39 2018

@author: Ben
"""

import ctypes
import numpy as np
import os
import signal
import sys
import time
import atsapi as ats
import gpib_control as gpib

siggen = gpib.connect(25)
t0 = time.time_ns()
siggen.write(':outp 1')
t1 = time.time_ns()
siggen.write(':outp 0')
t2 = time.time_ns()

print('Time to turn on: '+str(t1-t0)+' nanoseconds')
print('Time to turn off: '+str(t2-t1)+' nanoseconds')