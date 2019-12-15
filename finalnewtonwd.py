#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 22:44:29 2019

@author: baranhokelek
"""

import pandas
# I first tried using python's built-in csv library, but pandas seems to be way more convenient.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

G = 6.67408e-8 # cm^3 g^-1 s^2
solar_mass = 1.989e33 #g
earth_radius = 6.371e8 #cm

def mytests():
    
    ################## b ########################
    df = pandas.read_csv('white_dwarf_data.csv')
    
    # I use the formula: g = G*M/r^2
    df['logg'] = np.sqrt(G * solar_mass * df['mass'] / 10**df['logg'])/earth_radius
    
    # renaming of columns for understandability
    df.columns = ['WD ID', 'Radius(in Earth radius)', 'Mass(in solar mass)']
    
    # radius values weren't sorted before
    df = df.sort_values('Radius(in Earth radius)')

    plt.plot(df['Radius(in Earth radius)'], df['Mass(in solar mass)'], label='CSV data')
    plt.title("M vs R of Low-Temperature White Dwarfs")
    plt.xlabel("Radius(in Earth radius)")
    plt.ylabel("Mass(in solar mass)")
    plt.grid()
    
    #plt.savefig("WDMR.jpg", dpi=150)
    
    # I wanted to see if the curve looked the same when I plot these two quantities in CGS units
    # and to my surprise, it does. I don't know how an arbitrary scaling of variables(M0 for mass, Re for radius)
    # can result in a similar curve.
    # plt.figure()
    # plt.plot(df['Radius(in Earth radius)']*earth_radius, df['Mass(in solar mass)']*solar_mass)
    # plt.title("M vs R of Low-Temperature White Dwarfs")
    # plt.xlabel("Radius(in cm)")
    # plt.ylabel("Mass(in g)")
    # plt.grid()
    
    
    
    ################## c ########################
    # This is how we expect the M(R) function to behave, as proven in part a.
    def expected_curve_fun(R, C, n):
        return C * (R ** ((3-n)/(1-n)))
    
    true_rdata = df['Radius(in Earth radius)']
    true_mdata = df['Mass(in solar mass)']
    
    # I test the behaviour of the fit for various number of points to see which one looks better.
    # num_points array can be configured to give more plots for different values of n.
    num_points = np.arange(10, 100, 20)
    def plotz(n):
        
        fig, axs = plt.subplots(1, 2)
        
        rdata = df['Radius(in Earth radius)'][-1:-(n+1):-1]
        mdata = df['Mass(in solar mass)'][-1:-(n+1):-1]
        
        
        
        axs[0].plot(true_rdata, true_mdata, label='CSV')
        axs[0].set_title("M vs R")
        axs[0].set_xlabel("Radius(in Earth radius)")
        axs[0].set_ylabel("Mass(in solar mass)")
        axs[0].grid()
            
        popt,pcov = curve_fit(expected_curve_fun, rdata, mdata, bounds=([-np.inf, 1.01], [np.inf, np.inf]))
        print(popt)
        nstar = popt[1]
        q = 5*nstar/ (1 + nstar)
        axs[0].plot(rdata, expected_curve_fun(rdata, *popt), 'r', label="q=%.2f" % q)
        axs[0].plot(true_rdata, expected_curve_fun(true_rdata, *popt), 'r--')
        
        axs[0].set_xlim(0.3, 2.8)
        axs[0].set_ylim(0, 1.5)
        
        axs[0].legend(loc = 'upper right')
        
        
        axs[1].loglog(true_rdata, true_mdata, 'r', rdata, expected_curve_fun(rdata, *popt), 'b')
        axs[1].set_title("log-log CSV(r) vs %d points(b)" % n)
        axs[1].grid()
        
    for n in num_points:
        plotz(n),
    

    # This is the M(R) function once I fix q=3 (hence, n*=1.5 and 3-n/1-n = -3).
    def mass_q3(R, C):
        return C * (R ** -3)
    
    
    # This part assumes that q=2, and tries to fing a finer approximation to K*.
    
    fig, axs = plt.subplots(1, 2)  
    
    # In the previous part, n=50 gave the closest value to q(2.99), so I chose 50.
    n = 50
    rdata = df['Radius(in Earth radius)'][-1:-(n+1):-1]
    mdata = df['Mass(in solar mass)'][-1:-(n+1):-1]
    
    axs[0].plot(true_rdata, true_mdata, label='CSV')
    axs[0].set_title("M vs R")
    axs[0].set_xlabel("Radius(in Earth radius)")
    axs[0].set_ylabel("Mass(in solar mass)")
    axs[0].grid()
        
    popt,pcov = curve_fit(mass_q3, rdata, mdata)
    
    Kstar = popt[0]
    axs[0].plot(rdata, mass_q3(rdata, *popt), 'r', label="K*=%.2f" % Kstar)
    axs[0].plot(true_rdata, mass_q3(true_rdata, *popt), 'r--')
    
    axs[0].set_xlim(0.3, 2.8)
    axs[0].set_ylim(0, 1.5)
    
    #C = 0.7
    #n = 1.8
    #plt.plot(df['Radius(in Earth radius)'][-1:-(num_points+1):-1], C * (df['Radius(in Earth radius)'][-1:-(num_points+1):-1] ** ((3-n)/(1-n))))
    
    axs[0].legend(loc = 'upper right')
    
    
    axs[1].loglog(true_rdata, true_mdata, 'r', rdata, mass_q3(rdata, *popt), 'b')
    axs[1].set_title("log-log CSV(r) vs %d points(b)" % n)
    axs[1].grid()
    
    
    # When I set q to be 3, K* turns out to be approximately 2.
    
    plt.show()