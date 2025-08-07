# This program defines the units used in our program.
import numpy as np

import sys
import os
from scipy import interpolate
import matplotlib.pyplot as plt

datadir = os.path.realpath(__file__)[:-len('MyUnits.py')] + 'Data/'

# UNITS
# -----------------------------------------------

# Degrees (to avoid my own confusions)
rads = 1
degs = np.deg2rad(1.)

# Energy (in eV)
eV  = 1
MeV = 1e6*eV
GeV = 1e9*eV
TeV = 1e12*eV
PeV = 1e15*eV
EeV = 1e18*eV

# Mass (in grams)
g  = 1
kg = 1e3*g

# Length (in cm)
cm = 1
m  = 1e2*cm
km = 1e5*cm

# Time (in seconds)
s      = 1
minute = 60*s
hour   = 60*minute
day    = 24*hour
year   = 365.25*day

# Conversions in natural units
eV_to_s = 1.52e15 # 1 GeV = 1.52e24 s



# Physical constants
# --------------------------------------------------

c          = 299792.458*km/s  #light speed
ProtonMass = 1.6721e-27*kg

EarthRadius = 6371*km
RadioBend   = 1.13 # A factor to take into account the radio wave refraction

AirDensity = 1.340*kg/m**3
IceDensity = 0.916*g/cm**3
Avogadro   = 6.022e23

VolumePerParticle = 6.19e-25/cm**3

def DecayLength(lt):
    return c*lt

def MeanFreePath(xs):
    return VolumePerParticle/xs



# PREM data
# ---------------

def PREM_FullData():
    dat = np.loadtxt(datadir + 'PREM.dat')
    dat[:,0] = dat[:,0]*km
    dat[:,1] = dat[:,1]*g/cm**3
    return dat


def PREM_FullInterpolator():
    dat = PREM_FullData()
    f = interpolate.interp1d(dat[:,0],dat[:,1], kind = 'nearest', 
                                                fill_value= 'extrapolate', 
                                                bounds_error= False)
    return f


# STANDARD MODEL PARAMETERS
# --------------------------
def xsSM(enu):
    #enu in eV
    return 7.84e-36*cm**2 *(enu*eV/GeV)**0.36#3

def ltSM(enu,factor):
    #enu in eV
    # returns lt in s
    ldecay = 50*km*(enu/factor/EeV)
    return ldecay/c

# muon_stopping_data = np.loadtxt(datadir+"muB_rock_water.txt", skiprows=4)

# def mub_rock(emu):
#     return 3.55e-6*(np.log10(emu))**(1/6.5)*cm**2/g # heuristic fit, more or less valid for E > 1e4 GeV
#     return np.interp(np.log10(emu),np.log10(muon_stopping_data[:,0]*GeV),muon_stopping_data[:,1]*1e-6*cm**2/g)

# def mub_water(emu):
#     return 2.6e-6*(np.log10(emu))**(1/5.5) # heuristic fit, more or less valid for E > 1e4 GeV
#     return np.interp(np.log10(emu),np.log10(muon_stopping_data[:,0]*GeV),muon_stopping_data[:,2]*1e-6*cm**2/g)

# MuonDepth = 19*km # where does this come from? 
# mfpWater  = 550*km
# mfpEarth  = 205*km
# ctau_sm    = 8.380*km#T
# TauToMuBr = 0.1739

