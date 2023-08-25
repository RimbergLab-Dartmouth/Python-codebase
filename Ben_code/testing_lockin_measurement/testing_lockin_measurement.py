# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:12:19 2020

@author: Ben
"""

import instrument_classes_module as icm
import numpy as np
import matplotlib.pyplot as plt

lockin = icm.stanford_sr830(11)
LO_siggen = icm.agilent_e8257c(19)
RF_siggen = icm.keysight_n5183b(25)