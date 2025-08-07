# This file includes a collection of functions to compute
# quantities related to the KM3NeT experiment. 
# For instance: probabilities, effective areas, 
# number of events or likelihoods. 

import os
import sys

homedir = os.path.realpath(__file__)[:-len('Experiments/KM3NeT.py')]
sys.path.append(homedir)

datadir = homedir + 'Data/'

import Calculators.Probabilities as probs
import MyUnits as muns
import MyFunctions as mfuns
import CyFunctions as cfuns
import Parameters as pars
import PlottingVariables as plvars
import matplotlib.pyplot as plt

import numpy as np
from scipy import integrate
from scipy.interpolate import RectBivariateSpline

plvars.add_style(homedir+"PlotData/paper.mplstyle")

# ----------------------------------------- #
#               PARAMETERS                  #
# ----------------------------------------- #

Depth = 3.4*muns.km

SectionArea = 0.360*muns.km**2 # this adjusts the official KM3NeT effective area
Volume = 0.15*muns.km**3 #np.pi*IC_Radius**2*IC_Height
Radius = 0.3*muns.km # this is only useful for the decay of the T particle inside the detector
NumberOfTargets = muns.IceDensity*muns.Avogadro*Volume

LiveTime = 335*muns.day
Efficiency = 1.0 # This is adjusted to match the official KM3NeT effective area

Event_Azim = 259.8*muns.degs
Event_Elev = 0.6*muns.degs
Event_Err  = 1.5*muns.degs

Event_Energy = 120*muns.PeV
Event_EnergyErrorUp = 110*muns.PeV # let us assume a 40% precision (50PeV)
Event_EnergyErrorDown = 60*muns.PeV # let us assume a 40% precision (50PeV)
number_of_muons = 1

# let's say that we're interested in the muons with reconstructed energy between:
minEnergy = 10*muns.PeV 
maxEnergy = pars.y_NtoT*pars.y_Ttomu*pars.EnergyDelta

# Are cascades looked for in KM3NeT? if False, only tracks are counted
count_cascades = False 


############################################
#  Computing distances
############################################

# We build interpolators from the datafile which has 
# the water & rock distances in every direction of KM3NeT
distances = np.loadtxt(datadir+"KM3NeT/distances_from_ARCA.dat")
elev_angles = np.unique(distances[:,1])*muns.degs
azim_angles = np.unique(distances[:,0])*muns.degs
elevmin, elevmax = np.min(elev_angles), np.max(elev_angles)

# trajectories may cross up to two layers of water.
# in such case, A is the first one to be crossed, B the last one.
water_dataA = np.zeros((azim_angles.shape[0],elev_angles.shape[0]))
water_dataB = np.zeros((azim_angles.shape[0],elev_angles.shape[0]))
rock_data   = np.zeros((azim_angles.shape[0],elev_angles.shape[0]))

for i, ax in enumerate(azim_angles):
	for j, bx in enumerate(elev_angles):
		rock_data[i,j]   = distances[i*elev_angles.shape[0]+j,3]*muns.km
		water_dataA[i,j] = distances[i*elev_angles.shape[0]+j,2]*muns.km
		water_dataB[i,j] = distances[i*elev_angles.shape[0]+j,4]*muns.km

waterA_interp = RectBivariateSpline(azim_angles,elev_angles,water_dataA, kx = 1, ky = 1)
waterB_interp = RectBivariateSpline(azim_angles,elev_angles,water_dataB, kx = 1, ky = 1)
rock_interp   = RectBivariateSpline(azim_angles,elev_angles,rock_data, kx = 1, ky = 1)


def WaterRockDistances(elev,azimuth):
    """
    Returns the water and rock distances at the given elevation angle "elev"
    and azimuthal angle "azimuth" (0º = N), as seen from ARCA. 
    This takes into account the local topography around the detector. 
    Layers are ordered following the incoming particle trajectory,
    i.e. from the surface of water/rock to inside of KM3NeT.
    """
    return (waterB_interp.ev(azimuth, elev), 
            rock_interp.ev(azimuth,elev),
            waterA_interp.ev(azimuth,elev))

def Layers(azim,elev):
    """
    Layers are ordered from furthest away to closest away to the detector,
    i.e. following the incoming particle trajectory.
    """
    rho_w = 1.0*muns.g/muns.cm**3
    rho_e = 2.835*muns.g/muns.cm**3
    dK3 = Depth

    # elevmin sets the angle below which we use the PREM model
    if elev <= elevmin:
        ext = np.arcsin((1-dK3/muns.EarthRadius)*np.cos(elev))
        dist_to_exit = mfuns.K3_DistanceToExitPoint(elev,dK3)
        return cfuns.LayersThroughChord(ext,dist_to_exit)
    
    # above elevmax, the trajectory only crosses water, assume spherical Earth
    elif elev >= elevmax:
        dist_to_exit = mfuns.K3_DistanceToExitPoint(elev,dK3)
        ext = np.arcsin((1-dK3/muns.EarthRadius)*np.cos(elev))
        d = cfuns.ChordLength(ext) - dist_to_exit
        return np.array([[0,d,d,rho_w],
                         [0,d,d,rho_w]])
    
    else: # we do topography
        dwB, de, dwA = WaterRockDistances(elev,azim)

        dwB = dwB if dwB > 0.1 else 0.0 # to avoid some NaNs
        dwA = dwA if dwA > 0.1 else 0.0 
        de  = de  if de  > 0.1 else 0.0 

        if (dwB + de == 0):
            # the trajectory only goes through water
            # I duplicate it due to Cython stuff...
            return np.array([[0.0,dwA,dwA,rho_w],
                             [0.0,dwA,dwA,rho_w]])
        elif (dwB == 0):
            # the trajectory enters directly in rock
            return np.array([[0,de,de,rho_e],
                             [de,dwA+de,dwA,rho_w]])
        else: # the trajectory crosses two layers of water and one of rock
            return np.array([[0,dwB,dwB,rho_w],
                            [dwB,dwB+de,de,rho_e],
                            [dwB+de,dwA+dwB+de,dwA,rho_w]])
    
def LengthSmoothEarth(elev):
    """
    Returns the distance of the chord which crosses through ARCA at
    an elevation angle "elev". Assumes the Earth is spherical and
    smooth (without topography).
    """
    d = Depth
    ext = np.arcsin((1-d/muns.EarthRadius)*np.cos(elev))
    dist_to_exit = mfuns.K3_DistanceToExitPoint(elev,d)
    lay = cfuns.LayersThroughChord(ext,dist_to_exit)
    L = lay[-1,1] # total length travelled
    return L

def length(azim,elev,total_output = True):
    """ 
    Returns the total length in the (azim,elev) direction, either
    by topographic computation or by smooth Earth approximation.
    if total_output = True, return the total distance.
        Else, return a tuple with water and rock distance, respectively.
    """
    if (elev >= np.min(elev_angles)) and (elev <= np.max(elev_angles)):
        dwA, de, dwB = WaterRockDistances(elev,azim)
        if total_output == True:
            return dwA+de+dwB
        else:
            return dwA+dwB, de
    else:
        if total_output == True:
            return LengthSmoothEarth(elev)
        else:
            if elev > np.max(elev_angles):
                # All is water
                return LengthSmoothEarth(elev),0.0
            if elev < np.min(elev_angles):
                # All is rock
                return 0.0, LengthSmoothEarth(elev)
            
def column_depth(azim,elev):
    """ 
    Returns the total column depth in the (azim,elev) direction, either
    by topographic computation or by smooth Earth approximation.
    """
    rhor = 2.835*muns.g/muns.cm**3
    rhow = 1.000*muns.g/muns.cm**3
    if (elev >= np.min(elev_angles)) and (elev <= np.max(elev_angles)):
        dwA, de, dwB = WaterRockDistances(elev,azim)
        return dwA*rhow+de*rhor+dwB*rhow
    else:
        if elev > np.max(elev_angles):
            # All is water
            return LengthSmoothEarth(elev)*rhow
        if elev < np.min(elev_angles):
            # this is improbable, I should take into account the different densities of the Earth
            # All is rock
            return LengthSmoothEarth(elev)*rhor


############################################
# Computing probabilities
############################################

# In the following functions, einiN is the energy of the initial N particle.
# After scattering, the T particle leaves with energy einiN*pars.y_NtoT,
# and after decay, the muon leaves with energy einiN*pars.y_NtoT*pars.y_Ttomu,
# while the N leaves with energy einiN*pars.y_NtoT*pars.y_TtoN.

def ProbMuon_dmin(azim,elev,einiN,xs,lt):
    """
    Returns the probability that a muon arrives to the detector
    with energy larger than 10PeV, in the direction of (azim,elev),
    Check Parameters.py to activate/deactivate T absorption through pars.is_T_absorbed
    Implements equation (X.XX) from 2509.XXXXX.
    """
    lay = Layers(azim,elev) # This already takes into account the many possible ways to compute the layers: upgoing, downgoing, all.
    return probs.ProbMuon_dmin(lay,einiN*pars.y_NtoT*pars.y_Ttomu,xs,lt,pars.is_T_absorbed) # here enters the energy of the initial muon.

def ProbMuon(azim,elev, efin, einiN, xs,lt):
    """
    Returns the probability that a muon arrives to the detector
    with energy efin. This is not used in the final analysis.    
    """
    lay = Layers(azim,elev) # This already takes into account the many possible ways to compute the layers: upgoing, downgoing, all.
    return probs.ProbMuon(lay,efin,einiN,xs,lt,pars.is_T_absorbed)

def ProbN(azim,elev,efin,einiN,xs,lt):
    """
    Implements equation (A.19) from 2305.03746
    N is produced from T interaction (via xs) -one regeneration-.
    """
    lay = Layers(azim,elev) # This already takes into account the many possible ways to compute the layers: upgoing, downgoing, all.
    return probs.ProbN(lay,efin,einiN,xs,lt,pars.is_T_absorbed)

def ProbT(azim,elev,efin,einiN,xs,lt):
    """
    Probability a particle T exits the Earth with energy efin.
    Implements equation (A.21) from 2305.03746.
    """
    lay = Layers(azim,elev) # This already takes into account the many possible ways to compute the layers: upgoing, downgoing, all.
    return probs.ProbT(lay,efin,einiN,xs,lt,pars.is_T_absorbed)

def DecayProbabilityT(lt):
    """
    Probability a T particle decays inside the detector (once arrived).
    """
    return probs.ProbDecayIn(lt,2*Radius)

####################################
# -  Effective areas
####################################

def GeometricArea():
    """
    Returns the geometric area of the detector
    """
    return SectionArea

# In the following functions, pars.EnergyDelta is the energy of the initial flux of N particles.
# Then, in principle the events can have different final energies efin.

def EffectiveAreaMuon(azim, elev, xs, lt):
    """
    Returns the effective area to through-going muons,
    considering all muons with final energy above 10PeV.
    Implements equation (XX.X) from 2509.XXXXX
    """
    A_geom = GeometricArea()
    return Efficiency*A_geom*ProbMuon_dmin(azim,elev, pars.EnergyDelta, xs, lt)

def EffectiveAreaN(azim,elev,efin,xs,lt): 
    """
    Returns the effective area to T decays (and maybe T scattering too)    .
    Implements equation (XX.X) from 2509.XXXXX
    """
    return NumberOfTargets*xs*ProbN(azim,elev,efin,pars.EnergyDelta,xs,lt)*Efficiency

def EffectiveAreaT(azim,elev,efin,xs,lt):
    """
    Returns the effective area to N scattering.    
    Implements equation (XX.X) from 2509.XXXXX
    """
    probT = ProbT(azim,elev,efin,pars.EnergyDelta,xs,lt)
    area = GeometricArea()*DecayProbabilityT(lt)
    if pars.is_T_absorbed:
        area += NumberOfTargets*xs
    return area*probT*Efficiency

def EffectiveArea(azim,elev,efin, xs,lt):
    """
    Returns the total effective area in the direction of (azim,elev). 
    """
    return (EffectiveAreaMuon(azim,elev,  xs,lt) +
            EffectiveAreaN(azim,elev,efin,xs,lt) +
            EffectiveAreaT(azim,elev,efin,xs,lt))


def TotalEffectiveAreaMuon(xs,lt, eps = 1e-2):
    """
    Returns the total effective area to muons, all-sky integrated.
    """
    def ftopo(azim,elev):
        # for the values where azim is relevant (computing the topography)
        return np.cos(elev)*EffectiveAreaMuon(azim,elev,xs,lt)
    
    def fsmooth(elev):
        # for the values where azim is irrelevant (smooth Earth approximation)
        return np.cos(elev)*EffectiveAreaMuon(0,elev,xs,lt) # could have chosen whichever azimuth
    
    inttopo = integrate.dblquad(ftopo,elevmin,elevmax,0,2*np.pi,
                                     epsrel = eps)[0]
    intsmoo = 2*np.pi*(integrate.quad(fsmooth, elevmax,np.pi/2, epsrel = eps)[0]+
                       integrate.quad(fsmooth,-np.pi/2,elevmin, epsrel = eps)[0])
    
    return inttopo+intsmoo


def TotalEffectiveAreaN(xs,lt, eps = 1e-2):
    """
    Returns the total effective area to N, all-sky integrated.
    If our initial flux is a delta function in energies, we do not need an integration in energy (since initial and final energies are related by a delta)
    """
    def ftopo(azim,elev):
        return np.cos(elev)*(EffectiveAreaN(azim,elev,pars.EnergyDelta,xs,lt)+
                             EffectiveAreaN(azim,elev,pars.EnergyDelta*pars.y_NtoT*pars.y_TtoN,xs,lt))
    def fsmooth(elev):
        return np.cos(elev)*(EffectiveAreaN(0,elev,pars.EnergyDelta,xs,lt)+
                             EffectiveAreaN(0,elev,pars.EnergyDelta*pars.y_NtoT*pars.y_TtoN,xs,lt)) # could have chosen whichever azimuth
    
    inttopo, err = integrate.dblquad(ftopo,np.min(elev_angles),np.max(elev_angles),0,2*np.pi,
                                     epsrel = eps)
    intsmoo = 2*np.pi*(integrate.quad(fsmooth,np.max(elev_angles),np.pi/2,epsrel = eps)[0]+integrate.quad(fsmooth,-np.pi/2,np.min(elev_angles),epsrel = eps)[0])
    return inttopo+intsmoo

def TotalEffectiveAreaN1(xs,lt, eps = 1e-2):
    """
    Returns the total effective area to N, all-sky integrated.
    This function just takes into account N which have not been absorbed once.
    """
    def ftopo(azim,elev):
        return np.cos(elev)*(EffectiveAreaN(azim,elev,pars.EnergyDelta,xs,lt))
    def fsmooth(elev):
        return np.cos(elev)*(EffectiveAreaN(0,elev,pars.EnergyDelta,xs,lt))
    
    inttopo, err = integrate.dblquad(ftopo,np.min(elev_angles),np.max(elev_angles),0,2*np.pi,
                                     epsrel = eps)
    intsmoo = 2*np.pi*(integrate.quad(fsmooth,np.max(elev_angles),np.pi/2,epsrel = eps)[0]+integrate.quad(fsmooth,-np.pi/2,np.min(elev_angles),epsrel = eps)[0])
    return inttopo+intsmoo

def TotalEffectiveAreaN2(xs,lt, eps = 1e-2):
    """
    Returns the total effective area to N, all-sky integrated.
    This function just takes into account N which have been regenerated after T decay.
    """
    def ftopo(azim,elev):
        return np.cos(elev)*(EffectiveAreaN(azim,elev,pars.EnergyDelta*pars.y_NtoT*pars.y_TtoN,xs,lt))
    def fsmooth(elev):
        return np.cos(elev)*(EffectiveAreaN(0,elev,pars.EnergyDelta*pars.y_NtoT*pars.y_TtoN,xs,lt)) # could have chosen whichever azimuth
    
    inttopo, err = integrate.dblquad(ftopo,np.min(elev_angles),np.max(elev_angles),0,2*np.pi,
                                     epsrel = eps)
    intsmoo = 2*np.pi*(integrate.quad(fsmooth,np.max(elev_angles),np.pi/2,epsrel = eps)[0]+integrate.quad(fsmooth,-np.pi/2,np.min(elev_angles),epsrel = eps)[0])
    return inttopo+intsmoo

def TotalEffectiveAreaT(xs,lt, eps = 1e-1):
    """
    Returns the total effective area to T, all-sky integrated.
    """
    def ftopo(azim,elev):
        return np.cos(elev)*EffectiveAreaT(azim,elev,pars.EnergyDelta*pars.y_NtoT,xs,lt)
    def fsmooth(elev):
        return np.cos(elev)*EffectiveAreaT(0.0,elev,pars.EnergyDelta*pars.y_NtoT,xs,lt) # could have chosen whichever azimuth
    inttopo, err = integrate.dblquad(ftopo,np.min(elev_angles),np.max(elev_angles),0,2*np.pi,
                                     epsrel = eps)
    intsmoo = 2*np.pi*(integrate.quad(fsmooth,np.max(elev_angles),np.pi/2,epsrel = eps)[0]+integrate.quad(fsmooth,-np.pi/2,np.min(elev_angles),epsrel = eps)[0])
    return inttopo+intsmoo

def TotalEffectiveArea(xs,lt,br):
    """
    Returns the total effective area to all possible signals.
    Muons are only produced if T decays into a muon (given by br),
    while N/T scattering and T decay always produce a signal.
    """
    return br*TotalEffectiveAreaMuon(xs,lt)+TotalEffectiveAreaN(xs,lt)+TotalEffectiveAreaT(xs,lt)

def AverageEffectiveAreaMuon(xs,lt, azrec = Event_Azim,
                                    threc = Event_Elev,
                                    dth   = Event_Err,
                                    eps   = 1e-2):
    """
    Averages the effective area with a gaussian around the measured angle, with error dth.
    Implements equation (XX.X) from 2509.XXXXX
    """

    def f(th,az):
        # I am neglecting the covariance matrix of the angular error.
        intt  = np.cos(th)*EffectiveAreaMuon(az,th,xs,lt)
        intt *= cfuns.exp_weight(az,azrec,dth)
        intt *= cfuns.exp_weight(th,threc,dth)
        return intt

    # If one wants to improve the integral, probably one can just assume that it goes from
    # the measured point (azrec,threc) and 10sigma away from it (which is already a lot!)
    n = 5
    az_min, az_max = azrec - n*dth, azrec + n*dth
    th_min, th_max = threc - n*dth, threc + n*dth
    
    int, err = integrate.dblquad(f,az_min,az_max,th_min,th_max,epsrel=eps)
    return int

# -----------------------------------------
# Muon energy-dependent probability
# ----------------------------------------

# In this data release we have neglected this part of the code
# because it requires a heavy montecarlo file, which we have to decided
# to leave aside to keep the release light. 
# This code is not necessary for the analysis performed in our papers,
# only for some complementary figure in the appendix.
# Please contact the developers for further info on this code.

# muon_points = np.loadtxt(homedir+"Data/all_muon_data_530PeV.dat")
# muon_points[:,0] *= muns.PeV
# muon_points[:,1] *= muns.km
# xarr = np.unique(muon_points[:,1])

# ebins = np.geomspace(100*muns.TeV,530*muns.PeV,9)
# elow, eupp = ebins[:-1], ebins[1:]

# def ProbMuonTail(azim,elev,eebin,xs,lt):
#     lay = Layers(azim,elev) 
#     probhist = 0
#     for j, dmu in enumerate(xarr):
#         pt = probs.ProbMuonAtd(lay,dmu,xs,lt,pars.is_T_absorbed)
#         wj = 1*muns.km*pt
#         subarr = muon_points[np.where(muon_points[:,1]==dmu)]
#         # print("subarr", subarr)
#         Nj = subarr.shape[0]+1
#         # print(Nj)
#         pj = np.histogram(subarr[:,0],eebin)[0]/Nj
#         # print(wj,wj*pj)
#         probhist += wj*pj[0]
    
#     return probhist

# def efficiency_ic(ec):
#     # taken from 1502.02649
#     c, d, q = 0.5, 1.1, 4.6
#     fsat = 0.453844
#     e_th = 100*muns.TeV
#     x = np.log10(ec/e_th)
#     return (c*x**q/(1.+d*x**q))/fsat

# def EffectiveAreaMuonTail(azim,elev,eebin,xs,lt):
#     return ProbMuonTail(azim,elev,eebin,xs,lt)*Efficiency*efficiency_ic(np.sqrt(eebin[0]*eebin[1]))*GeometricArea()


# def TotalEffectiveAreaMuonTail(eebin,xs,lt,eps =1e-2):
#     def ftopo(azim,elev):
#         # for the values where azim is relevant (computing the topography)
#         # print(EffectiveAreaMuonTail(azim,elev,eebin,xs,lt))
#         return np.cos(elev)*EffectiveAreaMuonTail(azim,elev,eebin,xs,lt)
    
#     def fsmooth(elev):
#         # for the values where azim is irrelevant (smooth Earth approximation)
#         # I don't need to do a dblquad integral here :)
#         return np.cos(elev)*EffectiveAreaMuonTail(0,elev,eebin,xs,lt) # could have chosen whichever azimuth

#     inttopo = integrate.dblquad(ftopo,elevmin,elevmax,0,2*np.pi,
#                                      epsrel = eps)[0]
#     intsmoo = 2*np.pi*(integrate.quad(fsmooth, elevmax,np.pi/2, epsrel = eps)[0]+
#                        integrate.quad(fsmooth,-np.pi/2,elevmin, epsrel = eps)[0])
    
#     return inttopo+intsmoo

# ---------------------------------------------#
#               NUMBER OF EVENTS               #
# ---------------------------------------------#

def DiffuseEvents(xs, lt, br, 
                          phi = 1/muns.km**2/muns.day, 
                          DT = LiveTime,
                          return_sum = True):
    """
    Returns the total number of events assuming a diffuse flux.
    If return_sum = False, events are returned as
    [through-going muons, cascades, starting tracks]
    """    
    fnorm = pars.DiffuseNorm # This is because Phi is integrated for all solid angle, so avoid double-counting
    if return_sum:
        int_area = TotalEffectiveArea(xs,lt,br)
        return DT*phi*fnorm*int_area
    else:
        area_mu = TotalEffectiveAreaMuon(xs,lt)
        area_N  = TotalEffectiveAreaN(xs,lt)
        area_T  = TotalEffectiveAreaT(xs,lt)
        return np.array([DT*phi*fnorm*area_mu*br, # through-going muons
                         DT*phi*fnorm*(area_N+(1-br)*area_T), #cascades
                         DT*phi*fnorm*area_T*br]) # starting tracks

def BestFitFlux(xs,lt):
    DT = LiveTime
    nevents = 1
    return nevents/pars.DiffuseNorm/DT/TotalEffectiveArea(xs,lt)


# ---------------------------------------------#
#               INTERPOLATING                  #
# ---------------------------------------------#

# For speeding up computations, if one has already computed total and averaged
# effectives areas (see Calculators/EffectiveAreas.py), interpolating them from
# a data file speeds up the computation of tests statistics.

areatot_fname = datadir+'KM3NeT/EffectiveArea_TAbsorption.dat'
areaavg_fname = datadir+'KM3NeT/AveragedArea_TAbsorption.dat'

interp_aeff_mu   = mfuns.create_interpolator_from_datafile(areatot_fname, 2) # area to muons
interp_aeff_NT   = mfuns.create_interpolator_from_datafile(areatot_fname, (3,4)) # area to N+T
interp_aeff_T    = mfuns.create_interpolator_from_datafile(areatot_fname, 4) # area to T
interp_avgd_aeff = mfuns.create_interpolator_from_datafile(areaavg_fname, 2) # area to muons around the event

def reset_interpolators():
    # This will work only if areatot_fname and areaavg_fname are not rewritten!
    global interp_aeff_mu, interp_aeff_NT, interp_aeff_T, interp_avgd_aeff
    interp_aeff_mu   = mfuns.create_interpolator_from_datafile(areatot_fname, 2)
    interp_aeff_NT   = mfuns.create_interpolator_from_datafile(areatot_fname, (3,4))
    interp_aeff_T    = mfuns.create_interpolator_from_datafile(areatot_fname, 4)
    interp_avgd_aeff = mfuns.create_interpolator_from_datafile(areaavg_fname, 2)
    return
    
# ---------------------------------------------#
#               TEST STATISTICS                #
# ---------------------------------------------#

def DiffuseEventTestStatistic(xs,lt,br):
    """
    Computes the test statistic (TS) for ANITA given xs, lt,br;
    phi is already marginalized.
    Implements equation (XX.X) from 2509.XXXXX
        """

    if count_cascades and pars.N_detectable: # KM3NeT looks for cascades and they are detectable
        int_aeff = br*interp_aeff_mu.ev(xs,lt)  + interp_aeff_NT.ev(xs,lt)
    elif count_cascades: # KM3NeT looks for cascades but the primary vertex is not detectable
        int_aeff = br*interp_aeff_mu.ev(xs,lt)  + interp_aeff_T.ev(xs,lt)
    else: # KM3NeT does not look for cascades or they are not detectable
        int_aeff = br*(interp_aeff_mu.ev(xs,lt) + interp_aeff_T.ev(xs,lt))

    avg_aeff = br*interp_avgd_aeff.ev(xs,lt)

    chi2 = -2*np.log(avg_aeff/int_aeff)
    return chi2

def DiffuseEventTestStatisticFromPhi(xs,lt,br,phi):
    """
    Computes the test statistic (TS) for ANITA given xs, lt, br, phi;
    Implements equation (XX.X) from 2509.XXXXX. 
    """
    fnorm = pars.DiffuseNorm

    if count_cascades and pars.N_detectable: # KM3NeT looks for cascades and they are detectable
        int_aeff = br*interp_aeff_mu.ev(xs,lt) + interp_aeff_NT.ev(xs,lt)
    elif count_cascades: # KM3NeT looks for cascades but the primary vertex is not detectable
        int_aeff = br*interp_aeff_mu.ev(xs,lt) + interp_aeff_T.ev(xs,lt)
    else: # KM3NeT does not look for cascades or they are not detectable
        int_aeff = br*(interp_aeff_mu.ev(xs,lt) + interp_aeff_T.ev(xs,lt))

    avg_aeff = br*interp_avgd_aeff.ev(xs,lt)

    chi2  = -2*np.log(fnorm*phi*LiveTime*avg_aeff)
    chi2 += 2*fnorm*phi*LiveTime*int_aeff

    return chi2

