# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:18:31 2020

@author: Sisira
Module containing classes for instruments requiring gpib connection.
"""
import numpy as np

from instrument_classes_module import gpib_instrument as gpib

class SRS_844(gpib):
    def __init__(self, addr):
        """
        inherit all the methods of gpib_instrument class from
        instrument_classes_module
        """
        super().__init__(addr)
        
    def get_output_display(self):
        """
        Gets the current display.
        
        Returns:
            str: current display
        """
        
        chone_str=self.query('DDEF?1')
        chone_out=int(chone_str)
        chtwo_str=self.query('DDEF?2')
        chtwo_out=int(chtwo_str)
        chone_disp=['X','Rvolts','Rdbm']
        chtwo_disp=['Y','theta']
        return chone_disp[chone_out]+chtwo_disp[chtwo_out]
    
    def set_output_display(self,display='XY'):
        """
        Sets the display into (X,Y), (Rvolts,theta) or (Rdbm,theta)
        
        Parameters:
            display (str): 'XY','Rvolts','Rdbm'
         
         Returns:
            str: Current display 
        """
        
        if display=='XY':
            self.write('DDEF1,0')
            self.write('DDEF2,0')
            
        elif display=='Rvolts':
            self.write('DDEF1,1')
            self.write('DDEF2,1')
            
        elif display=='Rdbm':
            self.write('DDEF1,2')
            self.write('DDEF2,1')
            
        return self.get_output_display()
    
    def get_reference_phase(self):
        """
        Gets the reference phase in degrees.
        
        Returns:
            float: Phase reference in deg
        """
        
        phase_str=self.query('PHAS?')
        return float(phase_str)
    
    def set_reference_phase_deg(self,phase):
        """
        Sets the reference phase in degrees.
        
        Parameters:
            phase (float): The reference phase in degrees.
        
        Returns:
            float: the reference phase.
        """
        
        self.write('PHAS{}'.format(phase))
        return self.get_reference_phase()
    
    def get_sensitivity(self):
        """
        Gets the sensitivity.
        
        Returns:
            str: Value of sensitivity.
        """
        
        sens_index=self.query('SENS?')
        sens_str=['100 nVrms / -127 dBm', '300 nVrms / -117 dBm',
                  '1 µVrms / -107 dBm','3 µVrms / -97 dBm',
                  '10 µVrms / -87 dBm','30 µVrms / -77 dBm',
                  '100 µVrms / -67 dBm', '300 µVrms / -57 dBm',
                  '1 mVrms / -47 dBm','3 mVrms / -37 dBm',
                  '10 mVrms / -27 dBm','30 mVrms / -17 dBm',
                  '100 mVrms / -7 dBm','300 mVrms / +3 dBm',
                  '1 Vrms / +13 dBm']
        return sens_str[int(sens_index)]
    
    def set_sensitivity_nV(self,sens):
        """
        Sets the sensitivity.
        
        Parameters:
            sens(float): Sensitivity in nV.
            
        Return:
            str: Value of sensitivity.
        """
        
        sens_value=np.array([100,300,1e3,3e3,10e3,30e3,100e3,300e3,1e6,3e6,
                             10e6,30e6,100e6,300e6,1e9])
        sens_index=np.where(sens_value==sens)
        self.write('SENS{}'.format(sens_index[0][0]))
        return self.get_sensitivity()
    
    def get_time_constant(self):
        """
        Gets the time constant.
        
        Returns:
            str: Value of time constant.
        """
        
        tc_index=self.query('OFLT?')
        tc_str=['100 µs', '300 µs', '1 ms', '3 ms', '10 ms', '30 ms', '100 ms',
                '300 ms', '1 s', '3 s', '10 s', '30 s', '100 s', '300 s', 
                '1 ks', '3 ks', '10 ks', '30 ks']
        return tc_str[int(tc_index)]
    
    def set_time_constant_sec(self,tc):
        """
        Sets the time constant.
        
        Parameters:
            tc(float): Time constant in seconds.
            
        Returns:
            str: Value of time constant.
        """
        
        tc_value=np.array([100e-6,300e-6,1e-3,3e-3,10e-3,30e-3,100e-3,
                           300e-3,1,3,10,30,100,300,1e3,3e3,10e3,30e3])
        tc_index=np.where(tc_value==tc)
        self.write('OFLT{}'.format(tc_index[0][0]))
        return self.get_time_constant()
    
    def push_sensitivity(self,act):
        """
        Push sensitivity one up or down.
        
        Parameters:
            act (str): 'up' or 'down'.
            
        Returns:
            Value of sensitivity.
        """
        if act=='up':
            self.write('KEYP 4')
        if act=='down':
            self.write('KEYP 5')
        return self.get_sensitivity()
    
    def push_time_constant(self,act):
        """
        Push time constant one up or down.
        
        Parameters:
            act (str): 'up' or 'down'.
            
        Returns:
            Value of time constant.
        """
        if act=='up':
            self.write('KEYP 0')
        if act=='down':
            self.write('KEYP 1')
        return self.get_time_constant()
    