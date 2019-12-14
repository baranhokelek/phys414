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

G = 6.67408e-8 # cm^3 g^-1 s^2
solar_mass = 1.989e33 #g
earth_radius = 6.371e8 #cm

def mytests():
    df = pandas.read_csv('white_dwarf_data.csv')
    
    # I use the formula: g = G*M/r^2
    df['logg'] = np.sqrt(G * solar_mass * df['mass'] / 10**df['logg'])/earth_radius
    
    # renaming of columns for understandability
    df.columns = ['WD ID', 'Radius(in Earth radius)', 'Mass(in solar mass)']
    
    # radius values weren't sorted before
    df = df.sort_values('Radius(in Earth radius)')

    plt.plot(df['Radius(in Earth radius)'], df['Mass(in solar mass)'])
    plt.title("M vs R of Low-Temperature White Dwarfs")
    plt.xlabel("Radius(in Earth radius)")
    plt.ylabel("Mass(in solar mass)")
    plt.grid()
    
    plt.savefig("WDMR.jpg", dpi=150)
    
    # I wanted to see if the curve looked the same when I plot these two quantities in CGS units
    # and to my surprise, it does. I don't know how an arbitrary scaling of variables(M0 for mass, Re for radius)
    # can result in a similar curve.
    # plt.figure()
    # plt.plot(df['Radius(in Earth radius)']*earth_radius, df['Mass(in solar mass)']*solar_mass)
    # plt.title("M vs R of Low-Temperature White Dwarfs")
    # plt.xlabel("Radius(in cm)")
    # plt.ylabel("Mass(in g)")
    # plt.grid()